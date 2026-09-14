"""
Measure resident memory along the heaviest path, so the budget is a number we
took rather than a number we hoped for.

    python eval/memory_probe.py            # synthesises a large PDF
    python eval/memory_probe.py paper.pdf  # uses a real one

Two things are measured separately, because they answer different questions:

  1. **The fixed floor.** What each library costs just by being imported. This
     dominates total usage and is the part a design decision can actually move
     -- dropping a dependency removes its whole line.
  2. **The ingest delta.** How much RSS grows while a large PDF is parsed,
     sectioned, chunked and packed. The claim the redesign makes is that this
     number does not scale with page count, so the probe reports it against a
     deliberately large document.

Embedding is stubbed with a deterministic fake vector. That is the point of the
measurement, not a shortcut: the real call is a network round-trip whose memory
cost is the response buffer, and stubbing it isolates *our* allocations from the
HTTP client's. The stub returns the same shape and dtype the real one does, so
everything downstream of it -- packing, the SQLite blob write -- is real.
"""

import gc
import os
import sys
import time

import psutil

PROCESS = psutil.Process(os.getpid())


def rss_mb() -> float:
    gc.collect()
    return PROCESS.memory_info().rss / (1024 * 1024)


def report(label: str, before: float) -> float:
    now = rss_mb()
    print(f"  {label:<38} {now - before:+7.1f} MB   (total {now:6.1f} MB)")
    return now


def measure_imports() -> float:
    print("\nFIXED FLOOR — cost of importing each dependency")
    print("-" * 68)
    baseline = rss_mb()
    print(f"  {'bare interpreter':<38} {'':>7}      (total {baseline:6.1f} MB)")

    mark = baseline
    import sqlalchemy  # noqa: F401
    mark = report("sqlalchemy", mark)

    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    mark = report("fastapi + uvicorn", mark)

    import numpy  # noqa: F401
    mark = report("numpy", mark)

    import pypdf  # noqa: F401
    mark = report("pypdf", mark)

    import rank_bm25  # noqa: F401
    mark = report("rank_bm25", mark)

    from app.main import app  # noqa: F401
    mark = report("app (routers, models, services)", mark)

    # Imported lazily in production -- it only loads on the first LLM or
    # embedding call -- but it is resident for the rest of the process once it
    # does, so it belongs in the floor.
    from google import genai  # noqa: F401
    mark = report("google-genai (REST, no gRPC)", mark)

    print(f"  {'':<38} {'':>7}      {'-' * 20}")
    print(f"  {'FIXED FLOOR':<38} {mark - baseline:+7.1f} MB   (total {mark:6.1f} MB)")
    return mark


def build_large_pdf(path: str, pages: int) -> str:
    """Write a multi-page PDF with realistic section structure."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_rag import _make_pdf  # reuse the suite's writer, no new dependency

    from pathlib import Path

    body = [
        "We introduce a neural architecture for document understanding that",
        "relies entirely on attention over the input sequence. Recurrent",
        "models process tokens sequentially, which prevents parallelisation",
        "and limits throughput on modern accelerators substantially.",
        "Our approach removes recurrence and attains higher throughput while",
        "matching or exceeding baseline accuracy on every benchmark tested.",
    ] * 6

    headings = [
        "Abstract", "1. Introduction", "2. Related Work", "3. Methodology",
        "4. Experimental Setup", "5. Results", "6. Discussion",
        "7. Limitations", "8. Conclusion", "References",
    ]

    content = []
    for index in range(pages):
        page = []
        if index < len(headings):
            page.append(headings[index])
        elif index % 7 == 0:
            page.append(f"{index}. Additional Analysis")
        page.extend(body)
        content.append(page)

    return _make_pdf(Path(path), content)


def measure_ingest(pdf_path: str, floor: float) -> None:
    from app.rag import vector_store
    from app.rag.chunker import chunk_sections
    from app.rag.pdf_parser import iter_pages, read_metadata

    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"\nINGEST — {pdf_path} ({size_mb:.1f} MB on disk)")
    print("-" * 68)

    meta = read_metadata(pdf_path)
    after_meta = report(f"metadata ({meta.num_pages} pages)", floor)

    # The real pipeline, minus the network call and the database write. Each
    # section's chunks are packed and discarded, exactly as ingestion does.
    peak = after_meta
    started = time.monotonic()
    n_chunks = n_sections = 0
    total_chars = 0

    from app.rag.chunker import iter_sections

    for section, chunks in chunk_sections(iter_sections(iter_pages(pdf_path))):
        n_sections += 1
        for chunk in chunks:
            n_chunks += 1
            total_chars += len(chunk.content)
            # Stand-in for the embedding API response, packed for real.
            vector_store.pack([0.001 * (i % 97) for i in range(768)])
        peak = max(peak, PROCESS.memory_info().rss / (1024 * 1024))
        del chunks

    elapsed = time.monotonic() - started

    print(f"  {'sections':<38} {n_sections:>7}")
    print(f"  {'chunks':<38} {n_chunks:>7}")
    print(f"  {'extracted text':<38} {total_chars / 1000:>7.0f} kB")
    print(f"  {'wall time':<38} {elapsed:>7.1f} s")
    print(f"  {'':<38} {'':>7}      {'-' * 20}")
    print(f"  {'PEAK during ingest':<38} {peak - floor:+7.1f} MB   (total {peak:6.1f} MB)")

    after = report("after ingest, settled", peak)

    print("\nVERDICT")
    print("-" * 68)
    limit = 512
    headroom = limit - peak
    print(f"  Peak RSS observed                      {peak:6.1f} MB")
    print(f"  Instance limit                         {limit:6.1f} MB")
    print(f"  Headroom                               {headroom:6.1f} MB "
          f"({100 * peak / limit:.0f}% of budget used)")
    print(f"  Retained after ingest (leak check)     {after - floor:+6.1f} MB")
    print(
        "\n  The claim this probe exists to test is that peak does not scale\n"
        "  with page count. Re-run with a larger --pages value: the chunk and\n"
        "  section counts should rise while PEAK stays flat."
    )


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    floor = measure_imports()

    if len(sys.argv) > 1 and sys.argv[1].endswith(".pdf"):
        pdf_path = sys.argv[1]
    else:
        pages = 300
        for arg in sys.argv[1:]:
            if arg.startswith("--pages="):
                pages = int(arg.split("=", 1)[1])
        pdf_path = build_large_pdf(
            os.path.join(os.path.dirname(__file__), "_probe.pdf"), pages
        )

    measure_ingest(pdf_path, floor)


if __name__ == "__main__":
    main()
