"""
Step 5: hybrid retrieval -- dense (semantic) + BM25 (keyword), fused with RRF.

This is the most interesting file in the project, so here is the reasoning in
full.

Dense vector search understands *meaning*. Ask "how did they train it?" and it
finds "we optimised with AdamW for 40 epochs" with no shared words. But it is
weak on exact tokens: rare identifiers like "BERT-base", "CIFAR-100" or
"equation 4" get blurred into the surrounding semantics, so the exact chunk you
wanted can rank below vaguer ones.

BM25 is the opposite. Classic keyword scoring -- it rewards rare terms and nails
exact matches, but has no idea "training" and "optimisation" are related.

The two fail in *different* directions, which is exactly when combining helps.

Fusing them: Reciprocal Rank Fusion, scoring each document as
    sum over rankers of  1 / (k + rank)
with k = 60. The key property is that RRF uses only *ranks*, never raw scores.
Cosine similarity (0-1) and BM25 (unbounded) are not on a comparable scale, so
adding or averaging them would let BM25's larger numbers dominate. RRF sidesteps
normalisation entirely, and a chunk both rankers place near the top beats one
that only a single ranker loves.

Query expansion (new)
---------------------
The question is rewritten into a small number of variants before retrieval, and
every variant is run through both rankers. This attacks the failure mode neither
ranker fixes: the reader asks "is it fast?" and the passage says "inference
latency of 12ms". Dense search half-bridges that; a rewritten variant that
literally says "inference latency throughput" bridges it properly, and BM25 can
then match it exactly.

Each variant is its own ranked list going into the fusion, so a chunk that
several phrasings all surface rises -- which is the same argument as fusing two
rankers, applied to phrasings.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk
from app.rag import llm, prompts, vector_store
from app.rag.embeddings import EmbeddingError, embed_queries

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the original RRF paper; damps top-rank dominance


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    page_start: int
    page_end: int
    section: str | None
    score: float


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. BM25 needs pre-tokenised input.

    Hyphens and dots inside a token are kept, then also emitted split: a query
    for "BERT-base" should match "BERT base", and "F1-score" should match "F1".
    Model names and metric names are exactly the queries dense search is worst
    at, so BM25 needs to handle their punctuation properly to be worth having.
    """
    tokens: list[str] = []
    for match in re.findall(r"\b[\w][\w.\-]*\b", text.lower()):
        tokens.append(match)
        if "-" in match or "." in match:
            tokens.extend(part for part in re.split(r"[.\-]", match) if part)
    return tokens


def expand_query(question: str, paper_title: str, n_variants: int) -> list[str]:
    """Rewrite the question into alternative phrasings.

    The original question is always first in the returned list, so a failure
    here degrades to exactly the old behaviour rather than breaking retrieval.
    That matters because this runs on the latency path of every message: an LLM
    hiccup should cost the rewrites, not the answer.
    """
    if n_variants <= 1:
        return [question]

    try:
        raw = llm.generate(
            prompts.build_query_expansion_prompt(
                question, n_variants - 1, paper_title
            ),
            temperature=0.4,  # higher than elsewhere: we want genuinely different phrasings
            max_output_tokens=prompts.QUERY_EXPANSION_MAX_TOKENS,
        )
    except llm.LLMError as exc:
        logger.info("Query expansion unavailable, using the original only: %s", exc)
        return [question]

    variants = [question]
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*[-*\d.)\]]+\s*", "", line).strip()
        # Guard against the model returning commentary instead of queries.
        if 3 < len(cleaned) < 400 and cleaned.lower() != question.lower():
            variants.append(cleaned)

    return variants[:n_variants]


def _bm25_search(
    db: Session,
    queries: list[str],
    paper_id: str,
    top_k: int,
    section_id: str | None = None,
) -> list[list[RetrievedChunk]]:
    """Keyword search over one paper's chunks, one ranked list per query.

    The BM25 index is built once here and scored against every query variant,
    rather than rebuilt per variant. It is still rebuilt on every *request*: for
    a few hundred chunks that is a handful of milliseconds, and it keeps the
    index always consistent with the database with no invalidation logic. At
    larger scale you would move to Postgres full-text or OpenSearch -- a fair
    thing to say out loud rather than pretend this scales.
    """
    query = select(Chunk).where(Chunk.paper_id == paper_id)
    if section_id:
        query = query.where(Chunk.section_id == section_id)
    chunks = list(db.scalars(query.order_by(Chunk.chunk_index)))
    if not chunks:
        return []

    bm25 = BM25Okapi([_tokenize(c.content) for c in chunks])

    rankings: list[list[RetrievedChunk]] = []
    for query in queries:
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        rankings.append(
            [
                RetrievedChunk(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    score=float(score),
                )
                for chunk, score in ranked[:top_k]
                if score > 0  # zero means no query term appeared at all
            ]
        )
    return rankings


def _dense_search(
    db: Session,
    queries: list[str],
    paper_id: str,
    top_k: int,
    section_id: str | None = None,
) -> list[list[RetrievedChunk]]:
    """Semantic search, one ranked list per query variant.

    All variants are embedded in a single API call -- three round-trips on the
    latency path of every message would be plainly worse than one.
    """
    try:
        vectors = embed_queries(queries)
    except EmbeddingError as exc:
        # Degrade to keyword-only rather than failing the question. BM25 alone
        # is a worse retriever, not a broken one, and an answer grounded in
        # keyword hits beats a 503. This is the concrete reason `mode` and the
        # BM25 path are kept even though dense wins on our eval set.
        logger.warning("Dense retrieval unavailable, falling back to BM25: %s", exc)
        return []

    rankings: list[list[RetrievedChunk]] = []
    for vector in vectors:
        hits = vector_store.search(
            db, vector, paper_id=paper_id, top_k=top_k, section_id=section_id
        )
        rankings.append(
            [
                RetrievedChunk(
                    chunk_id=h.chunk_id,
                    content=h.content,
                    page_start=h.page_start,
                    page_end=h.page_end,
                    section=h.section,
                    score=h.score,
                )
                for h in hits
            ]
        )
    return rankings


def _reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], top_k: int
) -> list[RetrievedChunk]:
    """Merge several ranked lists into one using RRF."""
    fused_scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            by_id.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)

    results: list[RetrievedChunk] = []
    for chunk_id, score in ordered[:top_k]:
        chunk = by_id[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                score=round(score, 5),
            )
        )
    return results


def retrieve(
    db: Session,
    query: str,
    paper_id: str,
    top_k: int | None = None,
    mode: str = "hybrid",
    expand: bool = True,
    paper_title: str = "",
    section_id: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question.

    `mode` exists so the evaluation script can measure dense / bm25 / hybrid
    against the same question set. Being able to show a before-and-after table is
    worth more than asserting hybrid is better -- and on our eval set it is
    currently dense that wins, which the README says plainly rather than quietly
    keeping fusion and implying it helps.

    Deduplication is inherent: RRF keys on chunk_id, so a chunk surfaced by three
    variants and both rankers appears once, with a correspondingly higher score.

    `section_id` narrows both rankers to one section. A reader who knows the
    answer is in Results should be able to say so; it also shrinks the matrix
    and the BM25 corpus for that query, so it is cheaper, not just sharper.
    """
    top_k = top_k or settings.TOP_K

    queries = (
        expand_query(query, paper_title, settings.QUERY_VARIANTS)
        if expand and mode != "bm25"
        else [query]
    )

    # Over-fetch from each ranker so fusion has room to reorder before we cut.
    fetch_k = top_k * 2

    if mode == "dense":
        dense = _dense_search(db, queries, paper_id, fetch_k, section_id)
        return _reciprocal_rank_fusion(dense, top_k) if dense else []
    if mode == "bm25":
        keyword = _bm25_search(db, queries, paper_id, fetch_k, section_id)
        return _reciprocal_rank_fusion(keyword, top_k) if keyword else []

    dense = _dense_search(db, queries, paper_id, fetch_k, section_id)
    keyword = _bm25_search(db, queries, paper_id, fetch_k, section_id)

    logger.debug(
        "retrieval: %d variants, %d dense lists, %d bm25 lists",
        len(queries), len(dense), len(keyword),
    )
    return _reciprocal_rank_fusion(dense + keyword, top_k)
