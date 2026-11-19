"""Automated Retrieval & Grounding Evaluation Suite comparing BM25, Dense, and Hybrid modes."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src/ is importable when running `python eval/run_eval.py` directly
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_rag.config import Settings  # noqa: E402
from hybrid_rag.engine import HybridRAGEngine  # noqa: E402
from hybrid_rag.schemas import (  # noqa: E402
    EvalSummary,
    ModeEvalMetrics,
    QueryRequest,
    RetrievalMode,
)


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: set[str], k: int) -> float:
    """1.0 if any expected document appears in the top-k retrieved chunks, else 0.0."""
    if not expected_doc_ids:
        return 0.0
    top_docs = set(retrieved_doc_ids[:k])
    return 1.0 if (top_docs & expected_doc_ids) else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_ids: set[str]) -> float:
    """1 / rank of the first retrieved chunk belonging to an expected document."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: set[str],
    k: int,
    total_relevant_chunks: int | None = None,
) -> float:
    """Standard chunk-level binary NDCG@k over the top-k retrieved chunks."""
    if not expected_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevances = [1.0 if doc_id in expected_doc_ids else 0.0 for doc_id in top_k]
    dcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(relevances, start=1))

    ideal_k = min(k, total_relevant_chunks) if total_relevant_chunks is not None else k
    if ideal_k <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_k + 1))
    return dcg / idcg if idcg > 0 else 0.0


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected domain keywords present in the synthesized answer."""
    if not expected_keywords:
        return 1.0
    lower_ans = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_ans)
    return hits / len(expected_keywords)


async def run_benchmark(
    dataset_path: Path | None = None,
    corpus_dir: Path | None = None,
    output_path: Path | None = None,
) -> EvalSummary:
    """Run full benchmark across bm25, dense, and hybrid modes and save JSON report."""
    ds_path = dataset_path or (REPO_ROOT / "eval" / "qa_dataset.json")
    corp_dir = corpus_dir or (REPO_ROOT / "data" / "corpus")
    out_path = output_path or (REPO_ROOT / "eval" / "latest_report.json")

    dataset: list[dict[str, Any]] = json.loads(ds_path.read_text(encoding="utf-8"))

    # Use an isolated temporary index directory so eval never pollutes user runtime state
    with tempfile.TemporaryDirectory(prefix="hybrid_rag_eval_") as tmp_dir:
        settings = Settings(
            index_dir=Path(tmp_dir),
            corpus_dir=corp_dir,
            llm_provider="extractive-grounded",
            embedding_provider="local-svd",
            auto_load_corpus=False,
        )
        engine = HybridRAGEngine(settings=settings)
        engine.load_corpus_directory(corp_dir)
        doc_count, chunk_count = engine.counts()
        doc_chunk_counts = {d.doc_id: d.chunk_count for d in engine.list_documents()}

        modes: list[RetrievalMode] = ["bm25", "dense", "hybrid"]
        mode_summaries: dict[str, ModeEvalMetrics] = {}
        per_query_records: list[dict[str, Any]] = []

        # Initialize per-query record stubs
        for item in dataset:
            per_query_records.append(
                {
                    "id": item["id"],
                    "language": item["language"],
                    "category": item["category"],
                    "answerable": item["answerable"],
                    "query": item["query"],
                    "expected_doc_ids": item["expected_doc_ids"],
                    "results": {},
                }
            )

        for mode in modes:
            r3_scores: list[float] = []
            r5_scores: list[float] = []
            mrr_scores: list[float] = []
            ndcg5_scores: list[float] = []
            faith_scores: list[float] = []
            kw_scores: list[float] = []
            latencies_ms: list[float] = []

            tp_abstain = 0
            fp_abstain = 0
            tn_answer = 0
            fn_hallucinate = 0

            for q_idx, item in enumerate(dataset):
                resp = await engine.query(
                    QueryRequest(query=item["query"], mode=mode, top_k=5)
                )
                latencies_ms.append(resp.latency.total_ms)
                retrieved_docs = [h.chunk.doc_id for h in resp.hits]
                expected_docs = set(item["expected_doc_ids"])

                if item["answerable"]:
                    r3 = recall_at_k(retrieved_docs, expected_docs, 3)
                    r5 = recall_at_k(retrieved_docs, expected_docs, 5)
                    rr = reciprocal_rank(retrieved_docs, expected_docs)
                    total_rel = sum(doc_chunk_counts.get(d, 0) for d in expected_docs)
                    ndcg = ndcg_at_k(retrieved_docs, expected_docs, 5, total_relevant_chunks=total_rel)
                    kw = 0.0 if resp.abstained else keyword_recall(resp.answer, item["expected_keywords"])

                    r3_scores.append(r3)
                    r5_scores.append(r5)
                    mrr_scores.append(rr)
                    ndcg5_scores.append(ndcg)
                    kw_scores.append(kw)
                    if resp.abstained:
                        fp_abstain += 1
                    else:
                        tn_answer += 1
                        faith_scores.append(resp.faithfulness_score)
                else:
                    if resp.abstained:
                        tp_abstain += 1
                    else:
                        fn_hallucinate += 1

                per_query_records[q_idx]["results"][mode] = {
                    "top_docs": retrieved_docs[:5],
                    "abstained": resp.abstained,
                    "confidence_score": resp.confidence_score,
                    "faithfulness_score": resp.faithfulness_score,
                    "latency_ms": resp.latency.total_ms,
                }

            total_q = len(dataset)
            abstain_acc = (tp_abstain + tn_answer) / max(total_q, 1)
            abstain_prec = (
                tp_abstain / (tp_abstain + fp_abstain)
                if (tp_abstain + fp_abstain) > 0
                else 1.0
            )
            abstain_rec = (
                tp_abstain / (tp_abstain + fn_hallucinate)
                if (tp_abstain + fn_hallucinate) > 0
                else 1.0
            )

            lat_arr = np.asarray(latencies_ms, dtype=np.float64)
            mode_summaries[mode] = ModeEvalMetrics(
                mode=mode,
                recall_at_3=round(float(np.mean(r3_scores)) if r3_scores else 0.0, 4),
                recall_at_5=round(float(np.mean(r5_scores)) if r5_scores else 0.0, 4),
                mrr=round(float(np.mean(mrr_scores)) if mrr_scores else 0.0, 4),
                ndcg_at_5=round(float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0, 4),
                citation_faithfulness=round(
                    float(np.mean(faith_scores)) if faith_scores else 0.0, 4
                ),
                keyword_answer_recall=round(
                    float(np.mean(kw_scores)) if kw_scores else 0.0, 4
                ),
                abstention_accuracy=round(abstain_acc, 4),
                abstention_precision=round(abstain_prec, 4),
                abstention_recall=round(abstain_rec, 4),
                latency_p50_ms=round(float(np.percentile(lat_arr, 50)), 2),
                latency_p95_ms=round(float(np.percentile(lat_arr, 95)), 2),
            )

    answerable_cnt = sum(1 for item in dataset if item["answerable"])
    unanswerable_cnt = len(dataset) - answerable_cnt

    summary = EvalSummary(
        total_queries=len(dataset),
        answerable_queries=answerable_cnt,
        unanswerable_queries=unanswerable_cnt,
        corpus_documents=doc_count,
        corpus_chunks=chunk_count,
        modes=mode_summaries,
        per_query_results=per_query_records,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def format_markdown_table(summary: EvalSummary) -> str:
    """Format a clean GitHub-flavored Markdown comparison table."""
    lines = [
        f"### Hybrid RAG Eval Benchmark ({summary.total_queries} queries: "
        f"{summary.answerable_queries} answerable, {summary.unanswerable_queries} unanswerable | "
        f"{summary.corpus_documents} docs, {summary.corpus_chunks} chunks)",
        "",
        "| Mode | Recall@3 | Recall@5 | MRR | NDCG@5 | Citation Faithfulness | Keyword Recall | Abstention Acc | Latency p50 / p95 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode_key in ("bm25", "dense", "hybrid"):
        m = summary.modes[mode_key]
        label = "**HYBRID (RRF+Rerank)**" if mode_key == "hybrid" else mode_key.upper()
        lines.append(
            f"| {label} | {m.recall_at_3 * 100:.1f}% | {m.recall_at_5 * 100:.1f}% | "
            f"{m.mrr:.3f} | {m.ndcg_at_5:.3f} | {m.citation_faithfulness * 100:.1f}% | "
            f"{m.keyword_answer_recall * 100:.1f}% | {m.abstention_accuracy * 100:.1f}% | "
            f"{m.latency_p50_ms:.2f} / {m.latency_p95_ms:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    summary = asyncio.run(run_benchmark())
    print(format_markdown_table(summary))
    print(f"\nSaved full JSON report -> {REPO_ROOT / 'eval' / 'latest_report.json'}")


if __name__ == "__main__":
    main()
