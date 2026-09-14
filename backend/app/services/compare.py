"""
Multi-paper questions: "how does A's method differ from B's?"

The unconstrained design is a shared vector index over every paper a user
owns, one approximate-nearest-neighbour query returning the best passages from
anywhere, and a cross-encoder reranker on top. None of that fits in 512 MB,
and none of it is needed: the per-paper design chosen for the memory budget
is also the right shape for comparison.

    for each selected paper:
        retrieve top-k chunks for the question   (one small NumPy matrix each,
                                                  loaded and dropped in turn)
    label every excerpt with its paper ("A", "B", "C")
    one generation call, citations resolved per paper

Peak memory is one paper's matrix at a time -- the same as single-paper chat --
plus the prompt, which grows linearly with the number of papers. That is why
the count is capped at MAX_PAPERS: three papers at TOP_K excerpts each is
~30k tokens of context, which is as much as a single prompt should carry on a
flash-tier model, and more papers than a reader can hold in mind anyway.

Citations carry the paper letter in their label ("Paper B, pages 4-5") and the
paper id in the response, so the UI can jump into the right PDF. Markers are
global across papers: the model sees [1]..[n] numbered continuously, and the
same renumbering as single-paper chat applies.

Comparisons are not persisted as chat history. They are a lookup across
papers, and the per-paper conversation memory would be the wrong place to
put them.
"""

import logging
import string

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Paper
from app.rag import llm, prompts
from app.rag.retriever import RetrievedChunk, retrieve
from app.schemas import CompareCitation, CompareResponse
from app.services import chat_service

logger = logging.getLogger(__name__)

MAX_PAPERS = 3
# Fewer excerpts per paper than single-paper chat: the prompt has to hold
# several papers' worth, and comparison questions are about the headline
# differences, not the fine print.
PER_PAPER_K = 6


def load_papers(db: Session, user_id: str, paper_ids: list[str]) -> list[Paper]:
    """The requested papers, in the requested order, all owned and ready.

    Ownership is enforced here rather than by the OwnedPaper dependency
    because there are several ids; an id belonging to someone else is simply
    absent from the result, which the caller turns into a 404 -- the same
    "does not exist as far as you are concerned" answer single-paper routes
    give.
    """
    unique = list(dict.fromkeys(paper_ids))
    rows = {
        p.id: p
        for p in db.scalars(
            select(Paper).where(Paper.id.in_(unique), Paper.user_id == user_id)
        )
    }
    return [rows[pid] for pid in unique if pid in rows]


def compare(db: Session, papers: list[Paper], question: str) -> CompareResponse:
    labels = string.ascii_uppercase[: len(papers)]

    # Retrieve per paper. Each call loads that paper's vectors, scores them,
    # and lets them go before the next paper is touched.
    per_paper: list[tuple[str, Paper, list[RetrievedChunk]]] = []
    for label, paper in zip(labels, papers, strict=True):
        with llm.calling(paper.id, "compare"):
            chunks = retrieve(
                db,
                query=question,
                paper_id=paper.id,
                top_k=PER_PAPER_K,
                paper_title=paper.title,
                expand=False,  # one expansion call per paper would triple the latency
            )
        per_paper.append((label, paper, chunks))

    flat: list[RetrievedChunk] = []
    owner: list[tuple[str, Paper]] = []  # aligned with `flat`
    for label, paper, chunks in per_paper:
        for chunk in chunks:
            flat.append(chunk)
            owner.append((label, paper))

    if not flat:
        return CompareResponse(
            answer="None of the selected papers contain passages relevant to that question.",
            citations=[],
            grounded=False,
            papers=[{"label": l, "id": p.id, "title": p.title} for l, p, _ in per_paper],
        )

    excerpts = prompts.format_compare_excerpts(flat, [label for label, _ in owner])
    prompt = prompts.build_compare_prompt(
        question=question,
        excerpts=excerpts,
        papers=[(label, p.title, (p.abstract or "")[:1200]) for label, p, _ in per_paper],
    )

    with llm.calling(None, "compare"):
        raw = llm.generate(prompt, temperature=0.2, max_output_tokens=prompts.CHAT_MAX_TOKENS)

    if prompts.INSUFFICIENT_CONTEXT in raw:
        return CompareResponse(
            answer=chat_service.NOT_FOUND_MESSAGE,
            citations=[],
            grounded=False,
            papers=[{"label": l, "id": p.id, "title": p.title} for l, p, _ in per_paper],
        )

    answer, base_citations = chat_service._extract_citations(raw, flat)

    # Attach the paper each citation came from. _extract_citations returns
    # them in first-seen order with chunk ids; map back through `flat`.
    by_chunk = {
        chunk.chunk_id: (label, paper)
        for chunk, (label, paper) in zip(flat, owner, strict=True)
    }
    citations: list[CompareCitation] = []
    for citation in base_citations:
        label, paper = by_chunk[citation.chunk_id]
        citations.append(
            CompareCitation(
                **citation.model_dump(),
                paper_id=paper.id,
                paper_label=label,
                paper_title=paper.title,
            )
        )
        citations[-1].label = f"Paper {label}, {citation.label}"

    return CompareResponse(
        answer=answer,
        citations=citations,
        grounded=bool(citations),
        papers=[{"label": l, "id": p.id, "title": p.title} for l, p, _ in per_paper],
    )

