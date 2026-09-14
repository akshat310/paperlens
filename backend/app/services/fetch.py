"""
Ingest a paper from a URL instead of an upload.

Two things happen here that an upload does not need:

1. **The server makes an outbound request on the user's behalf.** That is the
   textbook shape of SSRF -- a user-supplied URL that the server fetches could
   point at `http://169.254.169.254/` (a cloud metadata endpoint) or at a
   service on the internal network. So the host is checked against a short
   allowlist *before* any connection is made, redirects are followed only
   within the same allowlist, and the response is streamed through the same
   size cap as an upload. There is no general "fetch any PDF" feature, on
   purpose.

2. **arXiv gives us exact metadata.** The arXiv Atom API returns the title,
   authors, abstract and date as structured fields. That beats the heuristics
   in pdf_parser.py -- which guess the title from the largest line on page
   one -- so when the source is arXiv, the parser's guess is overridden.

Memory: the download is streamed to disk in the same 1 MB blocks as an upload
(`ingestion.hash_and_save` does the writing), so the body is never held whole.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.services.ingestion import hash_and_save

logger = logging.getLogger(__name__)

# Hosts we are willing to fetch from. Everything else is refused before a
# socket is opened. arXiv hosts the PDFs; doi.org only redirects, and the
# redirect target must itself be on this list.
ALLOWED_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org", "doi.org", "dx.doi.org"}

USER_AGENT = "PaperLens/2.0 (+https://github.com/akshat310/paperlens)"
TIMEOUT = httpx.Timeout(20.0, read=60.0)

_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_OLD_ARXIV_ID = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


class FetchError(ValueError):
    """The URL was refused or the download failed. Message is user-facing."""


@dataclass
class ArxivMeta:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    year: str | None


def _check_host(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("Only http(s) URLs are accepted.")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise FetchError(
            "Only arxiv.org and doi.org links are accepted. Upload the PDF directly "
            "for anything else."
        )
    return host


def arxiv_id_from_url(url: str) -> str | None:
    """'https://arxiv.org/abs/2405.12345v2' -> '2405.12345' (version dropped)."""
    if "arxiv.org" not in url:
        return None
    match = _ARXIV_ID.search(url) or _OLD_ARXIV_ID.search(url)
    return match.group(1) if match else None


def arxiv_metadata(arxiv_id: str) -> ArxivMeta | None:
    """Title/authors/abstract from the arXiv Atom API. None on any failure --
    metadata is a nicety; the parser's heuristics remain as the fallback."""
    try:
        response = None
        for attempt in (1, 2):
            response = httpx.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
            )
            # arXiv asks clients for a 3-second gap between API calls and
            # answers 429 when two papers are fetched back to back. One polite
            # wait is worth it; more is not, since the parser's own guess is
            # an acceptable fallback.
            if response.status_code == 429 and attempt == 1:
                time.sleep(3.5)
                continue
            break
        response.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ET.fromstring(response.text).find("a:entry", ns)
        if entry is None:
            return None
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        if not title or title.lower() == "error":
            return None
        authors = ", ".join(
            " ".join((a.findtext("a:name", "", ns) or "").split())
            for a in entry.findall("a:author", ns)
        )
        abstract = " ".join((entry.findtext("a:summary", "", ns) or "").split())
        published = entry.findtext("a:published", "", ns) or ""
        return ArxivMeta(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors or "",
            abstract=abstract,
            year=published[:4] if published[:4].isdigit() else None,
        )
    except Exception as exc:  # noqa: BLE001 -- best effort, never fatal
        logger.info("arXiv metadata lookup failed for %s: %s", arxiv_id, exc)
        return None


def _pdf_url(url: str) -> str:
    """Map an arXiv abstract page to its PDF; leave everything else alone."""
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id and "arxiv.org" in url:
        return f"https://arxiv.org/pdf/{arxiv_id}"
    return url


def download_pdf(url: str, destination: Path, max_bytes: int) -> tuple[str, int]:
    """Stream a PDF from an allowed host to disk. Returns (sha256, size).

    Redirects are followed by hand rather than with `follow_redirects=True`,
    because every hop has to pass the same host check -- a doi.org link that
    redirects to an arbitrary publisher host is exactly the case the allowlist
    exists for.
    """
    _check_host(url)
    current = _pdf_url(url)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT) as client:
        for _ in range(5):
            _check_host(current)
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    current = str(response.url.join(location))
                    continue
                if response.status_code != 200:
                    raise FetchError(f"The server answered {response.status_code} for that link.")

                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    raise FetchError(
                        "That link did not return a PDF. For DOI links, the publisher "
                        "may not offer an open-access PDF -- upload the file instead."
                    )

                # Adapt the byte iterator to the .read(n) interface that the
                # upload path expects, so one function writes both.
                return hash_and_save(_Reader(response.iter_bytes()), destination, max_bytes)

    raise FetchError("Too many redirects.")


class _Reader:
    """Wrap an iterator of byte blocks as a file-like object with .read()."""

    def __init__(self, blocks):
        self._blocks = blocks
        self._buffer = b""

    def read(self, n: int) -> bytes:
        while len(self._buffer) < n:
            try:
                self._buffer += next(self._blocks)
            except StopIteration:
                break
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out
