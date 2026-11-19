# Internal Engineering Guide: Chunking, Hybrid Retrieval, and RAG Evaluation Best Practices

## Why Structure-Aware Chunking Beats Fixed-Size Splitting
Naive fixed-character chunking frequently slices mid-sentence, separates headings from their body paragraphs, and destroys markdown tables or policy enumerations. Our production standard is **Structure-Aware Recursive Chunking**:
1. Parse document structure first (Markdown headings `#`, `##`, `###` or PDF page boundaries) and preserve the full hierarchical section breadcrumb in chunk metadata.
2. Split long sections recursively along natural boundaries: paragraphs (`\n\n`), then sentences, then word boundaries.
3. Target **400 to 800 characters** (roughly 100–200 tokens) per chunk with **10% to 15% overlap** (typically 60–100 characters) so facts spanning adjacent paragraphs remain retrievable without inflating context noise.

## Hybrid Retrieval: Combining BM25 and Dense Vectors
Single-mode retrieval consistently exhibits complementary blind spots:
- **Okapi BM25 (`k1=1.5, b=0.75`)** excels at exact lexical queries: regulation numbers (`UU No. 27 Tahun 2022`), error codes (`AI-SEV1`), acronyms (`DPO`, `PSI`, `KMS`), and rare proper nouns. However, BM25 struggles when users paraphrase concepts or use synonyms not verbatim in the text.
- **Dense Vector Retrieval** (L2-normalized semantic embeddings or subword/SVD latent vectors with cosine similarity) captures conceptual similarity, morphological variants, and paraphrased queries, but can drift on exact numeric thresholds or rare alphanumeric identifiers.

Combining both retrievers via **Reciprocal Rank Fusion (RRF)** eliminates score-calibration brittleness across different distributions:
`RRF(d) = sum_i 1 / (k + rank_i(d))` with default smoothing constant **`k = 60`**. RRF rewards documents that appear consistently high across both lexical and semantic rankers while preserving high-rank exact matches from either channel.

## Second-Stage Cross-Signal Reranking
After fusing the top 20–50 candidates via RRF, a second-stage reranker sharpens precision at the top of the list (`Top-3` and `Top-5`) before synthesis. Effective reranking features include:
- Normalized RRF consensus score.
- IDF-weighted query-term coverage (rewarding chunks covering multiple distinct query concepts rather than repeating one common word).
- Exact phrase and bigram match boost.
- Title and section heading alignment boost.
- Term proximity (tighter token span between matched query terms indicates direct discussion of the question).

## Grounding, Inline Citations, and Automatic Abstention
Hallucination is the single highest risk in enterprise RAG. Production pipelines must enforce three safeguards:
- **Strict Inline Citations**: Every factual claim must cite its source chunk explicitly using `[doc_id#chunk_id]`.
- **Claim-Level Faithfulness Verification**: Post-generation verification parses each claim sentence and checks lexical/semantic entailment against the cited chunk text, flagging unsupported assertions.
- **Automatic Abstention**: When top reranked evidence confidence or query-term support falls below the calibrated threshold, the engine must refuse to speculate and return an explicit abstention response (`abstained=True`). Never allow the generator to fall back on parametric pretraining memory for private enterprise questions.

## Quantitative RAG Evaluation Metrics
Never optimize RAG parameters by gut feel. Every retrieval or prompt change must be benchmarked against a version-controlled golden dataset (`eval/qa_dataset.json`) tracking:
- **Recall@3 and Recall@5**: Fraction of answerable queries where at least one ground-truth source document appears in the top-k retrieved results.
- **MRR (Mean Reciprocal Rank)**: Measures how early the first relevant document appears (`1 / rank_first_relevant`).
- **NDCG@5**: Rewards placing relevant documents higher within the top-5 list using logarithmic rank discounting.
- **Citation Faithfulness**: Average claim-to-citation textual support score across generated answers.
- **Abstention Accuracy**: Combined accuracy on correctly answering in-domain queries while cleanly abstaining on out-of-domain / unanswerable queries.
- **Latency p50 / p95**: Tracked per retrieval mode to validate performance budgets.
