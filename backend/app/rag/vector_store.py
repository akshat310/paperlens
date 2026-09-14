"""
Step 4: store and search vectors -- in SQLite, with NumPy doing the arithmetic.

There is no vector database in this process. That is the point of the file, so
here is the reasoning in full.

A vector database earns its keep when you are searching millions of vectors
across many documents and need an approximate index (HNSW, IVF) to avoid
scanning them all. **We are never in that situation.** Every search is scoped to
one paper, because answering a question about paper A must never surface a
passage from paper B. One paper is a few hundred chunks. Exhaustively scoring
400 vectors of 768 dimensions is 300k multiply-adds -- roughly 0.2 ms in NumPy,
and it is *exact*, where an ANN index is approximate. The index would make it
slower, not faster, once you count building and loading it.

What the database cost instead was memory: ChromaDB kept a Rust core, a client,
and an HNSW index resident for the life of the process. On a 512 MB instance
that is the single largest thing we could delete.

So: embeddings are `float32` blobs in the `chunks.embedding` column, alongside
the text they belong to. One store, no join across two systems that can
disagree, and deleting a paper's rows deletes its vectors -- the old design had
a real failure mode where SQL rows and Chroma vectors could drift apart if one
delete succeeded and the other did not.

**The vectors are unit length.** `embeddings._normalise` guarantees it. That is
what lets cosine similarity be a plain dot product here, with no division. If
that invariant is ever broken this file silently returns wrong rankings, which
is why it is asserted rather than assumed.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk

logger = logging.getLogger(__name__)

# The on-disk format. Little-endian float32, EMBEDDING_DIM values per chunk.
# Pinned explicitly rather than using the platform default so a database file
# written on one machine reads correctly on another.
DTYPE = np.dtype("<f4")


@dataclass
class SearchHit:
    chunk_id: str
    content: str
    page_start: int
    page_end: int
    section: str | None
    score: float  # cosine similarity, -1..1; in practice 0..1 for text


def pack(vector: list[float]) -> bytes:
    """Serialise one embedding for storage.

    float32, not float64. These are unit vectors with components around 1e-2,
    and float32 carries ~7 significant digits -- orders of magnitude more
    precision than a similarity ranking can use. It also halves both the row
    size and the array we build at query time.
    """
    return np.asarray(vector, dtype=DTYPE).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


def load_matrix(
    db: Session, paper_id: str, section_id: str | None = None
) -> tuple[list[str], np.ndarray]:
    """Load one paper's vectors as a single (n, dim) array.

    Returns (chunk_ids, matrix) with rows aligned to ids. Chunks whose embedding
    is null -- an embedding call that failed -- are skipped rather than
    zero-filled: a zero row scores 0 against everything and would quietly pad
    the result set with junk.

    Memory: 400 chunks x 768 dims x 4 bytes is 1.2 MB. It is built per query and
    dropped immediately after, which is affordable precisely because it is small.
    Caching it across requests would trade that 1.2 MB for a cache that has to
    be invalidated on every ingest, and would hold memory for every paper any
    user has ever opened.
    """
    query = select(Chunk.id, Chunk.embedding).where(
        Chunk.paper_id == paper_id, Chunk.embedding.is_not(None)
    )
    if section_id:
        query = query.where(Chunk.section_id == section_id)
    rows = db.execute(query.order_by(Chunk.chunk_index)).all()

    if not rows:
        return [], np.empty((0, settings.EMBEDDING_DIM), dtype=DTYPE)

    ids = [row[0] for row in rows]
    # np.frombuffer gives read-only views over the SQLite buffers; vstack copies
    # them into one contiguous array, which is what makes the dot product fast.
    matrix = np.vstack([unpack(row[1]) for row in rows])

    if matrix.shape[1] != settings.EMBEDDING_DIM:
        # Dimension mismatch means EMBEDDING_DIM changed after these rows were
        # written. Failing loudly beats returning nonsense rankings.
        raise ValueError(
            f"Stored embeddings are {matrix.shape[1]}-dimensional but "
            f"EMBEDDING_DIM is {settings.EMBEDDING_DIM}. The index was built with "
            "a different model or dimension -- re-ingest this paper."
        )

    return ids, matrix


def search(
    db: Session,
    query_embedding: list[float],
    paper_id: str,
    top_k: int = 10,
    section_id: str | None = None,
) -> list[SearchHit]:
    """Exact nearest-neighbour search within one paper.

    One dot product, one argpartition, one small sort. `argpartition` rather
    than a full `argsort` because we only need the top k in order, not all n --
    O(n) instead of O(n log n). At n=400 the difference is unmeasurable; it is
    written this way because it is the correct shape of the operation and costs
    nothing to get right.
    """
    ids, matrix = load_matrix(db, paper_id, section_id)
    if not ids:
        return []

    query = np.asarray(query_embedding, dtype=DTYPE)
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Query embedding has {query.shape[0]} dimensions, index has "
            f"{matrix.shape[1]}."
        )

    # Both sides are unit vectors, so the dot product *is* cosine similarity.
    scores = matrix @ query

    k = min(top_k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    # One query for the text of just the winners, rather than having carried
    # every chunk's content through the vector load above.
    winners = [ids[i] for i in top]
    by_id = {
        chunk.id: chunk
        for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(winners)))
    }

    hits: list[SearchHit] = []
    for index in top:
        chunk = by_id.get(ids[index])
        if chunk is None:  # deleted between the two queries; skip it
            continue
        hits.append(
            SearchHit(
                chunk_id=chunk.id,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                score=float(scores[index]),
            )
        )

    return hits
