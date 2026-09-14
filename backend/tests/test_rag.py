"""
Tests for the RAG pipeline's pure logic: parsing, sectioning, chunking, vector
search, and rank fusion.

These never touch the network or an API key. The PDF fixture is built by hand
with a tiny writer rather than by a library, so the suite has no dependency the
app itself does not have, and the vector tests use synthetic embeddings.
"""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Chunk, Paper, User
from app.rag import vector_store
from app.rag.chunker import (
    _overlap_tail,
    _split_sentences,
    chunk_sections,
    iter_sections,
)
from app.rag.pdf_parser import PageText, _classify_heading, iter_pages, read_metadata
from app.rag.retriever import RetrievedChunk, _reciprocal_rank_fusion, _tokenize


# ---------------------------------------------------------------------------
# A minimal PDF writer, so the fixture needs no extra dependency
# ---------------------------------------------------------------------------

def _make_pdf(path, pages: list[list[str]]) -> str:
    """Write a PDF whose pages contain the given lines of text."""
    objects: list[bytes] = []

    def add(body: bytes) -> None:
        objects.append(body)

    n = len(pages)
    page_ids = [4 + i * 2 for i in range(n)]

    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(
        f"<< /Type /Pages /Count {n} /Kids [".encode()
        + b" ".join(f"{p} 0 R".encode() for p in page_ids)
        + b"] >>"
    )
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, lines in enumerate(pages):
        commands = []
        y = 780
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            commands.append(f"BT /F1 11 Tf 1 0 0 1 56 {y} Tm ({escaped}) Tj ET")
            y -= 16
        content = "\n".join(commands).encode("latin-1", "replace")
        add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {page_ids[index] + 1} 0 R >>".encode()
        )
        add(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))
    return str(path)


BODY = [
    "We introduce a neural architecture for document understanding that",
    "relies entirely on attention. Recurrent models process tokens",
    "sequentially, which prevents parallelisation across the sequence.",
    "Our approach removes recurrence and attains higher throughput.",
]


@pytest.fixture()
def sample_pdf(tmp_path):
    return _make_pdf(
        tmp_path / "sample.pdf",
        [
            ["Attention Is Sufficient For Retrieval", "Ada Lovelace, Alan Turing",
             "Abstract", *BODY],
            ["1. Introduction", *BODY, *BODY],
            ["3. Methodology", *BODY, *BODY],
            ["5. Results", *BODY, "Table 1: accuracy of 94.2 percent on CIFAR-100."],
        ],
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_iter_pages_yields_every_page_with_numbers(sample_pdf):
    pages = list(iter_pages(sample_pdf))
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert all(p.text.strip() for p in pages)


def test_iter_pages_is_lazy():
    """The generator must not have read anything before it is iterated.

    This is the memory property the whole ingest path depends on, so it is
    asserted rather than assumed.
    """
    import inspect
    assert inspect.isgeneratorfunction(iter_pages)


def test_metadata_extracts_title(sample_pdf):
    meta = read_metadata(sample_pdf)
    assert "Attention Is Sufficient" in meta.title
    assert meta.num_pages == 4


def test_headings_are_classified_by_kind():
    assert _classify_heading("Abstract") == ("abstract", "Abstract")
    assert _classify_heading("3. Methodology")[0] == "method"
    assert _classify_heading("5 Results")[0] == "results"
    assert _classify_heading("Related Work")[0] == "related_work"
    assert _classify_heading("4 Sparse Attention Kernels")[0] == "other"


def test_prose_is_not_mistaken_for_a_heading():
    """A false heading mislabels every chunk after it, so this must be strict."""
    assert _classify_heading("In this introduction we survey prior results.") is None
    assert _classify_heading("The results, which we discuss below, are strong.") is None
    assert _classify_heading("") is None
    assert _classify_heading("x" * 200) is None


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------

def test_sections_group_pages_and_track_page_ranges(sample_pdf):
    sections = list(iter_sections(iter_pages(sample_pdf)))
    kinds = [s.kind for s in sections]

    assert "abstract" in kinds
    assert "method" in kinds
    assert "results" in kinds
    assert all(s.page_start <= s.page_end for s in sections)
    assert [s.ordinal for s in sections] == list(range(1, len(sections) + 1))


def test_text_before_any_heading_becomes_front_matter():
    """Text must never be dropped for want of a recognised heading."""
    pages = [PageText(page_number=1, text="Some unlabelled opening text.", headings=[])]
    sections = list(iter_sections(pages))
    assert len(sections) == 1
    assert sections[0].kind == "front_matter"
    assert "unlabelled opening" in sections[0].text


def test_unconventional_headings_still_keep_all_text():
    """A paper whose headings we do not recognise loses nothing, only labels."""
    pages = [
        PageText(page_number=1, text="Preamble. " * 20, headings=[]),
        PageText(page_number=2, text="More body text. " * 20, headings=[]),
    ]
    sections = list(iter_sections(pages))
    combined = " ".join(s.text for s in sections)
    assert "Preamble" in combined
    assert "More body text" in combined


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunks_carry_pages_sections_and_are_ordered(sample_pdf):
    pairs = list(chunk_sections(iter_sections(iter_pages(sample_pdf))))
    chunks = [c for _, cs in pairs for c in cs]

    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.page_start >= 1 and c.page_end >= c.page_start for c in chunks)
    assert all(c.section_name for c in chunks)


def test_chunks_respect_the_token_budget():
    long_text = "This is a sentence about attention mechanisms. " * 400
    pages = [PageText(page_number=1, text=long_text, headings=[])]
    pairs = list(chunk_sections(iter_sections(pages), target_tokens=200, overlap_tokens=20))
    chunks = [c for _, cs in pairs for c in cs]

    assert len(chunks) > 1
    # target + overlap, plus slack for not splitting the final sentence.
    budget = (200 + 20) * 4 + 200
    assert all(len(c.content) <= budget for c in chunks)


def test_consecutive_chunks_overlap():
    long_text = "Sentence number one about attention. Another sentence follows here. " * 120
    pages = [PageText(page_number=1, text=long_text, headings=[])]
    pairs = list(chunk_sections(iter_sections(pages), target_tokens=150, overlap_tokens=40))
    chunks = [c for _, cs in pairs for c in cs]

    assert len(chunks) >= 2
    tail = set(chunks[0].content.split()[-10:])
    head = set(chunks[1].content.split()[:25])
    assert tail & head, "consecutive chunks share no text"


def test_sentence_split_survives_academic_abbreviations():
    text = "We follow Vaswani et al. and use AdamW. Results are in Fig. 3 and Tab. 2."
    parts = _split_sentences(text)
    assert len(parts) == 2
    assert parts[0].endswith("AdamW.")


def test_overlap_starts_at_a_sentence_boundary():
    text = "First sentence here. Second sentence follows on. Third one closes it."
    tail = _overlap_tail(text, 40)
    assert tail[0].isupper()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _unit(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return (array / np.linalg.norm(array)).tolist()


def test_pack_unpack_roundtrips_within_float32_precision():
    vector = _unit([0.1, -0.2, 0.35, 0.9])
    restored = vector_store.unpack(vector_store.pack(vector))
    assert np.allclose(restored, vector, atol=1e-6)


def test_search_ranks_by_cosine_similarity(db, monkeypatch):
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 3)

    user = User(email="a@b.c", hashed_password="x")
    db.add(user)
    db.flush()
    paper = Paper(user_id=user.id, title="T", filename="f.pdf", file_path="/tmp/f.pdf")
    db.add(paper)
    db.flush()

    vectors = {
        "exact": _unit([1.0, 0.0, 0.0]),
        "close": _unit([0.9, 0.4, 0.0]),
        "far": _unit([0.0, 0.0, 1.0]),
    }
    for index, (name, vector) in enumerate(vectors.items()):
        db.add(
            Chunk(
                paper_id=paper.id, content=name, page_start=1, page_end=1,
                chunk_index=index, embedding=vector_store.pack(vector),
            )
        )
    db.commit()

    hits = vector_store.search(db, _unit([1.0, 0.0, 0.0]), paper.id, top_k=3)
    assert [h.content for h in hits] == ["exact", "close", "far"]
    assert hits[0].score > hits[1].score > hits[2].score


def test_search_skips_chunks_with_no_embedding(db, monkeypatch):
    """A failed embedding must not pad the results with a zero-scoring chunk."""
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 3)

    user = User(email="d@e.f", hashed_password="x")
    db.add(user)
    db.flush()
    paper = Paper(user_id=user.id, title="T", filename="f.pdf", file_path="/tmp/f.pdf")
    db.add(paper)
    db.flush()

    db.add(Chunk(paper_id=paper.id, content="has vector", page_start=1, page_end=1,
                 chunk_index=0, embedding=vector_store.pack(_unit([1.0, 0.0, 0.0]))))
    db.add(Chunk(paper_id=paper.id, content="no vector", page_start=1, page_end=1,
                 chunk_index=1, embedding=None))
    db.commit()

    hits = vector_store.search(db, _unit([1.0, 0.0, 0.0]), paper.id, top_k=5)
    assert [h.content for h in hits] == ["has vector"]


def test_search_on_an_empty_paper_returns_nothing(db):
    assert vector_store.search(db, [0.0] * 768, "no-such-paper", top_k=5) == []


def test_dimension_mismatch_raises_rather_than_ranking_nonsense(db, monkeypatch):
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 3)
    user = User(email="g@h.i", hashed_password="x")
    db.add(user)
    db.flush()
    paper = Paper(user_id=user.id, title="T", filename="f.pdf", file_path="/tmp/f.pdf")
    db.add(paper)
    db.flush()
    db.add(Chunk(paper_id=paper.id, content="c", page_start=1, page_end=1,
                 chunk_index=0, embedding=vector_store.pack([1.0, 0.0, 0.0])))
    db.commit()

    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 768)
    with pytest.raises(ValueError, match="different model or dimension"):
        vector_store.load_matrix(db, paper.id)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_tokenizer_splits_model_and_metric_names():
    """Exact identifiers are what BM25 is in the system to catch."""
    tokens = _tokenize("BERT-base scored F1-score 0.91")
    assert "bert-base" in tokens
    assert "bert" in tokens and "base" in tokens
    assert "f1" in tokens


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, content=chunk_id, page_start=1, page_end=1,
        section=None, score=0.0,
    )


def test_rrf_prefers_chunks_ranked_well_by_both_rankers():
    """The core property of RRF: broad agreement beats a single ranker's favourite."""
    dense = [_chunk("A"), _chunk("B"), _chunk("C")]
    bm25 = [_chunk("C"), _chunk("B"), _chunk("D")]

    fused = _reciprocal_rank_fusion([dense, bm25], top_k=4)
    ids = [c.chunk_id for c in fused]

    # With k=60:
    #   A = 1/61        = 0.016393
    #   B = 1/62 + 1/62 = 0.032258
    #   C = 1/63 + 1/61 = 0.032266   <- edges out B
    # Appearing in both rankings matters more than topping a single one.
    assert set(ids[:2]) == {"B", "C"}
    assert ids.index("A") > 1
    assert set(ids) == {"A", "B", "C", "D"}  # union, deduplicated


def test_rrf_deduplicates_across_query_variants():
    """Query expansion feeds several lists in; a chunk must still appear once."""
    variant_a = [_chunk("A"), _chunk("B")]
    variant_b = [_chunk("A"), _chunk("C")]
    variant_c = [_chunk("A")]

    fused = _reciprocal_rank_fusion([variant_a, variant_b, variant_c], top_k=5)
    ids = [c.chunk_id for c in fused]
    assert ids.count("A") == 1
    assert ids[0] == "A"  # surfaced by every variant, so it wins


def test_rrf_scores_are_descending():
    fused = _reciprocal_rank_fusion([[_chunk("A"), _chunk("B")], [_chunk("B")]], top_k=2)
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


# ---------- typography as a heading signal ----------

def test_year_numbered_bibliography_line_is_not_a_heading():
    from app.rag.pdf_parser import _classify_heading

    assert _classify_heading("2018. Contextual string embeddings for sequence labeling") is None
    assert _classify_heading("3.2 Attention") is not None


def test_lowercase_vocabulary_word_alone_is_not_a_heading():
    from app.rag.pdf_parser import _classify_heading

    assert _classify_heading("approach.") is None
    assert _classify_heading("Approach") == ("method", "Approach")


def test_bold_title_shaped_line_is_a_styled_heading():
    from app.rag.pdf_parser import _LineStyle, _styled_heading

    body = 10.0
    assert _styled_heading("3.2.1 Scaled Dot-Product Attention", _LineStyle("", 10.0, True), body)
    assert _styled_heading("Sparse Attention Kernels", _LineStyle("", 12.0, False), body)
    # Same size, not bold: body text.
    assert not _styled_heading("Sparse Attention Kernels", _LineStyle("", 10.0, False), body)
    # A single bold word is a table header, not a section.
    assert not _styled_heading("Model", _LineStyle("", 10.0, True), body)
    # A sentence, however bold, is not a heading.
    assert not _styled_heading(
        "The encoder is composed of a stack of six identical layers, each of which.",
        _LineStyle("", 10.0, True), body,
    )


def test_wrapped_heading_lines_are_merged_into_one_section():
    from app.rag.chunker import iter_sections
    from app.rag.pdf_parser import PageText

    page = PageText(
        page_number=1,
        text="BERT: Pre-training of Deep\nBidirectional Transformers\n\nWe introduce BERT.",
        headings=[
            ("other", "BERT: Pre-training of Deep", 0),
            ("other", "Bidirectional Transformers", 27),
        ],
    )
    sections = list(iter_sections([page]))
    assert len(sections) == 1
    assert sections[0].name == "BERT: Pre-training of Deep Bidirectional Transformers"
    assert "We introduce BERT." in sections[0].text
