"""
Build the fixture paper the evaluation runs against.

Why a generated paper rather than a real one committed to the repo: the eval
needs *ground truth* -- for every question, which page genuinely holds the
answer. Hand-labelling a real PDF is slow and the labels are only as good as my
reading of it; generating the document means the page of every fact is known by
construction, and anyone who clones the repo reproduces the identical corpus
without a licensing question.

The honest trade-off, which belongs in the README: a synthetic paper is cleaner
than real research prose, so absolute scores here are optimistic. The *relative*
comparison between retrieval modes is what this harness is for, and that stays
meaningful because all three modes see exactly the same corpus.

Page length matters more than it looks. An early version of this file used short
pages and produced a corpus of only 12 chunks, so retrieving the top 5 handed
back nearly half the document and every mode scored 100% -- a saturated
benchmark that measures nothing. Each page therefore carries a realistic volume
of prose, including material that is topically adjacent to other pages so that
the retriever has genuine opportunities to be wrong.

Every fact that a question asks about appears on exactly one designated page (or
the small set listed in questions.json). The surrounding prose is deliberately
written *around* those facts without restating them, because a fact duplicated
on a second page would silently invalidate the ground-truth labels.

Run:  python eval/make_fixture_paper.py
"""

import sys
from pathlib import Path

OUTPUT_PATH = "eval/fixture_paper.pdf"
TITLE = "RAGDoc: Retrieval-Augmented Question Answering over Scientific Documents"

# One entry per page. Page numbers are 1-based and match questions.json exactly.
#
# The content is written so the retrieval modes are genuinely distinguishable:
# some facts are stated in distinctive rare tokens ("HotpotQA", "3e-4", "A100")
# which BM25 should nail, and others are phrased so that answering requires
# matching meaning rather than words, which is where dense retrieval wins.
PAGES: list[tuple[str, str]] = [
    (
        "Abstract",
        "We present RAGDoc, a retrieval-augmented architecture for question "
        "answering over scientific documents. Existing systems either retrieve "
        "with sparse lexical matching, which fails on paraphrase, or with dense "
        "embeddings, which blur rare identifiers. RAGDoc combines both and "
        "attributes every generated claim to a specific source passage. On our "
        "primary benchmark RAGDoc reaches 84.2 F1, an improvement of 6.4 points "
        "over the strongest baseline. We release code and trained checkpoints. "
        "The remainder of this abstract sketches the shape of the contribution. "
        "Document question answering has matured rapidly, yet deployment in "
        "settings where correctness is checked remains rare, and practitioners "
        "report that the gap is one of verification rather than fluency. Our "
        "design begins from the assumption that a reader will want to confirm "
        "any claim before relying on it, and that the cost of confirmation must "
        "therefore be close to zero. Every component described in the following "
        "sections is shaped by that requirement, including the segmentation "
        "strategy, the retrieval stack, and the generation contract. We report "
        "results on three public corpora and one internally constructed set, "
        "and we include an ablation that isolates the contribution of each "
        "retrieval component. We further discuss the conditions under which the "
        "system declines to answer, which we regard as a feature of the design "
        "rather than a shortcoming to be minimised. Reproduction instructions, "
        "evaluation scripts, and the exact question sets used throughout are "
        "distributed alongside the implementation.",
    ),
    (
        "1. Introduction",
        "Researchers increasingly need to interrogate long technical documents "
        "rather than read them end to end. The central obstacle is trust: a "
        "language model asked about a document will readily produce a fluent "
        "answer that the document does not support. We argue that attribution, "
        "not fluency, is the binding constraint on usefulness. A system that "
        "declines to answer is more valuable than one that guesses, because a "
        "wrong answer with a confident tone costs the reader more time than no "
        "answer at all. RAGDoc is built around that principle. "
        "The intuition is easiest to see from the reader's position. Presented "
        "with a paragraph of confident prose, a reader has no way to separate "
        "the portion grounded in the source from the portion supplied by the "
        "model's own parameters. Verification then requires rereading the entire "
        "document, which is precisely the labour the system was meant to remove. "
        "The value of the tool collapses to zero, or below it, once the reader "
        "learns from experience that spot checks sometimes fail. "
        "We therefore treat the unit of output not as a paragraph but as a claim "
        "paired with a pointer. This reframing has consequences throughout the "
        "pipeline, and much of the engineering described later exists to keep "
        "that pointer trustworthy under conditions where it would otherwise "
        "degrade. The contributions of this work are an architecture that "
        "enforces the pairing, an evaluation methodology for measuring whether "
        "the pairing holds, and an empirical study of where it breaks.",
    ),
    (
        "2. Related Work",
        "Sparse retrieval with BM25 remains a strong baseline and is difficult "
        "to beat on queries containing rare terms. Dense passage retrieval, "
        "introduced by Karpukhin et al., encodes queries and passages into a "
        "shared vector space and substantially improves recall on paraphrased "
        "queries. ColBERT proposes late interaction as a middle ground. Fusion "
        "methods including Reciprocal Rank Fusion combine ranked lists without "
        "requiring score normalisation across systems. "
        "A parallel line of work studies attribution directly. Early systems "
        "appended a bibliography of consulted sources without indicating which "
        "source supported which sentence, an arrangement that offers the "
        "appearance of verifiability without its substance. Later work moves to "
        "sentence-level attribution and reports that models frequently attach "
        "citations to claims the cited passage does not contain. "
        "Work on abstention is comparatively sparse. Calibration studies show "
        "that model confidence correlates only loosely with correctness in "
        "open-ended generation, which suggests that thresholding on confidence "
        "is an unreliable route to declining gracefully. Approaches based on "
        "verifying candidate answers against retrieved evidence appear more "
        "robust, and our design follows that tradition. We differ principally "
        "in enforcing the constraint structurally rather than statistically, so "
        "that an unsupported claim is prevented from being emitted rather than "
        "detected after the fact.",
    ),
    (
        "3. Method Overview",
        "RAGDoc operates in three stages. First, a document is segmented into "
        "passages that never span a page boundary, so that every passage carries "
        "exactly one page number. Second, a hybrid retriever scores passages "
        "against the query using both a lexical and a semantic ranker, and the "
        "two ranked lists are merged. Third, a generator is conditioned on the "
        "merged passages and is instructed to cite the passages it used. "
        "The staging is deliberate. Each stage produces an artefact that can be "
        "inspected on its own, which matters because failures in this class of "
        "system are otherwise difficult to localise. When an answer is wrong, "
        "the question of whether the evidence was absent from the retrieved set "
        "or present but ignored has a definite answer, and that answer is "
        "recoverable without instrumenting the model itself. "
        "Stage boundaries are also where the guarantees live. The segmentation "
        "stage guarantees a property about page attribution. The retrieval stage "
        "guarantees that scoring is reproducible given a fixed index. The "
        "generation stage guarantees that any emitted pointer refers to a "
        "passage that was genuinely supplied. None of these guarantees depends "
        "on the behaviour of the language model, which is the reason they hold "
        "under distribution shift and across model versions.",
    ),
    (
        "4. Retrieval and Fusion",
        "The lexical ranker uses Okapi BM25 over tokenised passage text. The "
        "semantic ranker embeds passages with a sentence encoder producing "
        "384-dimensional vectors and retrieves by cosine similarity. The two "
        "lists are merged with Reciprocal Rank Fusion using a constant k of 60. "
        "We adopt rank-based fusion because cosine similarity is bounded between "
        "zero and one while BM25 scores are unbounded, so combining the raw "
        "values would allow one ranker to dominate the other arbitrarily. "
        "The two rankers fail in different directions, which is the condition "
        "under which combining them is worthwhile. Combining rankers that fail "
        "together yields the same errors at greater expense. We verified the "
        "independence assumption by measuring the overlap between the two "
        "candidate sets and found it substantially below what correlated "
        "rankers would produce. "
        "Fusion operates on ordinal position alone and therefore discards "
        "magnitude information, which is a genuine cost: a passage that one "
        "ranker considers overwhelmingly better than the next receives no more "
        "credit than a passage that barely leads. We accept this because the "
        "alternative requires calibrating two scoring functions against each "
        "other, and that calibration proved unstable across documents of "
        "differing length in preliminary experiments.",
    ),
    (
        "5. Passage Segmentation",
        "Passages are capped at 900 characters with an overlap of 150 characters "
        "between consecutive passages. The overlap ensures that a sentence "
        "spanning a segmentation boundary still appears intact in at least one "
        "passage. Segmentation is applied within a page rather than across the "
        "document, which costs a small amount of packing efficiency in exchange "
        "for unambiguous page attribution. "
        "Boundaries are chosen by descending separator preference. Paragraph "
        "breaks are preferred, then sentence terminators, then clause "
        "boundaries, and finally an arbitrary cut when no better option exists "
        "within the budget. This ordering keeps semantically coherent units "
        "intact where the text permits it. "
        "We examined finer and coarser settings. Shorter passages sharpen "
        "retrieval precision but frequently sever the connection between a "
        "claim and the qualifying clause that constrains it, producing passages "
        "that are individually retrievable and jointly misleading. Longer "
        "passages preserve context at the cost of diluting the retrievable "
        "signal, since a single relevant sentence is averaged against "
        "surrounding material during encoding. The chosen configuration sits at "
        "the point where both effects were acceptable.",
    ),
    (
        "6. Training Setup",
        "The generator is fine-tuned with the AdamW optimiser at a learning rate "
        "of 3e-4 for 40 epochs with a batch size of 32. Training was performed on "
        "eight NVIDIA A100 GPUs and completed in roughly 19 hours. We apply a "
        "linear warmup over the first 500 steps followed by cosine decay, and "
        "clip gradients at a norm of 1.0. Mixed precision training is used "
        "throughout. "
        "Supervision pairs each question with a target response containing "
        "pointers to supporting passages. Distractor passages are included in "
        "every training instance so that the model cannot succeed by attending "
        "indiscriminately to everything supplied. Roughly one instance in six is "
        "constructed to be unanswerable from the accompanying passages, with the "
        "target response being a refusal; without this proportion of negative "
        "examples the model learns that a plausible answer always exists. "
        "Checkpoints were selected on a held-out split using a criterion that "
        "combined answer quality with pointer correctness rather than answer "
        "quality alone. Selecting on answer quality in isolation reliably "
        "produced checkpoints that scored well while attaching pointers "
        "carelessly, which is the precise failure the architecture exists to "
        "prevent.",
    ),
    (
        "7. Datasets",
        "We evaluate on three question answering corpora. SQuAD provides "
        "extractive questions over single passages. Natural Questions contains "
        "genuine search queries paired with long documents. HotpotQA requires "
        "combining evidence from more than one passage to reach an answer. We "
        "additionally construct an internal set of 1,200 questions written by "
        "domain experts over 40 machine learning papers. "
        "The internal set exists because the public corpora share a property "
        "that flatters this class of system: their questions were written by "
        "annotators looking at the passage that answers them, which produces "
        "unusually high lexical overlap between question and evidence. Questions "
        "arising from genuine curiosity have no such guarantee, and the gap "
        "between the two regimes is large enough to change conclusions. "
        "Annotators for the internal set were instructed to write questions "
        "before locating the answer, and to include questions the document does "
        "not address. The latter category comprises approximately one fifth of "
        "the set and exists to measure whether the system declines when it "
        "should. Inter-annotator agreement on answerability was substantial, and "
        "disagreements were adjudicated by a third annotator.",
    ),
    (
        "8. Results",
        "RAGDoc attains 84.2 F1 on our primary benchmark, compared with 77.8 F1 "
        "for the dense-only baseline and 74.1 F1 for the lexical-only baseline. "
        "Attribution accuracy, measured as the proportion of generated claims "
        "traceable to a cited passage, reaches 91.6 percent. The system abstains "
        "on 7.3 percent of questions, and manual inspection confirms that the "
        "overwhelming majority of abstentions are cases the document genuinely "
        "does not address. "
        "The margin narrows considerably on the internally constructed set, "
        "which we regard as the more informative comparison for the reasons "
        "given in the preceding section. Both baselines degrade more sharply "
        "than the full system when question and evidence share little "
        "vocabulary, and the ordering of the two baselines reverses on that "
        "slice. "
        "Latency is dominated by generation rather than retrieval. Retrieval "
        "over a single document completes in a few tens of milliseconds and is "
        "not the bottleneck at any document length we examined. Memory scales "
        "linearly in the number of passages, and a document of two hundred "
        "pages remains comfortably within the working set of a single "
        "commodity machine.",
    ),
    (
        "9. Ablation Study",
        "Removing the lexical ranker and retrieving with embeddings alone costs "
        "3.1 F1, with the loss concentrated on questions containing rare "
        "identifiers such as dataset names and equation references. Removing the "
        "semantic ranker instead costs 5.7 F1, concentrated on paraphrased "
        "questions sharing no vocabulary with the source passage. Reducing the "
        "number of retrieved passages from five to two costs 4.2 F1. "
        "The asymmetry between the two ranker ablations is worth dwelling on, "
        "because it is easy to misread. It does not establish that semantic "
        "matching is the more important mechanism in general. It reflects the "
        "composition of the evaluation set, in which paraphrased questions "
        "outnumber those turning on a rare token. A corpus dense in "
        "identifiers, model names, or numeric constants would shift the balance, "
        "and we observed exactly that shift on the subset of our internal "
        "questions concerning experimental configuration. "
        "Removing the distractor passages from training, while leaving the "
        "architecture unchanged, costs 2.4 F1 and degrades attribution accuracy "
        "considerably more than it degrades answer quality. This supports the "
        "view that attribution and answer quality are separable capabilities "
        "that require separate supervision.",
    ),
    (
        "10. Limitations",
        "RAGDoc assumes that the source document yields clean extractable text. "
        "Scanned documents and pages where the content is presented as an image "
        "are not handled, since no optical character recognition step is "
        "performed. Tables and mathematical notation are linearised into plain "
        "text, which loses structure and occasionally corrupts the meaning of a "
        "numeric result. The system is monolingual and has only been evaluated "
        "on English text. "
        "Attribution is verified at the level of the passage rather than the "
        "sentence. A pointer therefore establishes that supporting material "
        "occurs somewhere within a region of up to nine hundred characters, "
        "which is a weaker guarantee than a reader might assume on first "
        "encountering the interface. "
        "Finally, the evaluation measures faithfulness to retrieved passages "
        "and not truth. A document containing an error will yield answers that "
        "faithfully reproduce that error, and the system will attach a pointer "
        "to it with full confidence. Nothing in this architecture detects a "
        "document that is internally consistent and wrong, and we caution "
        "against interpreting an attributed answer as a verified one.",
    ),
    (
        "11. Future Work",
        "We intend to add an optical character recognition stage so that scanned "
        "documents can be processed. A second direction is multi-document "
        "retrieval, allowing a question to be answered across a collection of "
        "related papers rather than a single one. We are also investigating "
        "query rewriting so that conversational follow-up questions can be "
        "resolved against earlier turns in a dialogue. "
        "Sentence-level attribution is the most direct response to the "
        "granularity limitation discussed above, and a promising route runs "
        "through a lightweight verification pass that aligns each emitted claim "
        "against spans within its cited passage. Preliminary experiments "
        "suggest the additional latency is tolerable when the pass runs only "
        "over claims the generator has already committed to. "
        "A more speculative direction concerns documents that disagree with one "
        "another. Once retrieval spans a collection, conflicting evidence "
        "becomes the normal case rather than an anomaly, and presenting a "
        "single answer misrepresents the state of the literature. We consider "
        "surfacing the disagreement to be the correct behaviour, though the "
        "interface question is unresolved.",
    ),
]


MAX_LINES_PER_PAGE = 62  # at 9pt with 13pt leading, between y=120 and y=750


def _wrap(text: str, chars_per_line: int = 95) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= chars_per_line:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build(path: str = OUTPUT_PATH) -> str:
    """Write the fixture PDF. Deterministic: same bytes on every run.

    Written with the minimal PDF writer from the test suite rather than a
    library. This used to use PyMuPDF, which the app no longer depends on --
    keeping a fixture generator that needs a package the application itself
    dropped would mean anyone regenerating it has to install a dependency for
    that one purpose.
    """
    backend = Path(__file__).resolve().parent.parent
    # Both paths: `tests` to import the module, and `backend` because test_rag
    # imports `app` at module level.
    sys.path.insert(0, str(backend))
    sys.path.insert(0, str(backend / "tests"))
    from test_rag import _make_pdf

    pages: list[list[str]] = []
    for page_number, (heading, body) in enumerate(PAGES, start=1):
        lines = [TITLE, "", heading, ""]
        lines.extend(_wrap(body))
        lines.append("")
        lines.append(str(page_number))

        # Text that does not fit is text that is silently absent from the
        # corpus, which would mean a labelled question whose answer is not
        # actually there -- quietly corrupting every metric this fixture feeds.
        # Fail loudly instead.
        if len(lines) > MAX_LINES_PER_PAGE:
            raise ValueError(
                f"Page {page_number} ({heading}) needs {len(lines)} lines but only "
                f"{MAX_LINES_PER_PAGE} fit. Shorten the body."
            )
        pages.append(lines)

    return _make_pdf(Path(path), pages)


if __name__ == "__main__":
    written = build()
    print(f"Wrote {written} ({len(PAGES)} pages)")
