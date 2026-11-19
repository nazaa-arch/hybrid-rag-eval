"""Strict citation grounding, claim-level faithfulness verification, and automatic abstention."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hybrid_rag.schemas import Citation, RetrievalHit
from hybrid_rag.text_utils import (
    char_ngrams,
    content_tokens,
    extract_bigrams,
    normalize_text,
    split_sentences,
)

_CITATION_TAG_RE = re.compile(r"\[([a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+)\]")
_NUMERIC_QUERY_RE = re.compile(
    r"\b(berapa|batas|maksimal|minimal|lama|durasi|persen|denda|sanksi|jam|hari|tahun|"
    r"threshold|thresholds|sla|slo|latency|p50|p95|p99|how many|how long|what is the|percentage|ratio|budget)\b",
    re.IGNORECASE,
)
_HAS_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?")


@dataclass
class AbstentionDecision:
    """Result of the confidence and abstention gate."""

    should_abstain: bool
    confidence_score: float
    reason: str | None = None


@dataclass
class VerificationResult:
    """Result of claim-level citation faithfulness verification."""

    faithfulness_score: float
    supported_claims: int
    total_claims: int
    citations: list[Citation]


class GroundingEngine:
    """Computes evidence confidence, synthesizes grounded answers, and verifies citations."""

    def __init__(
        self,
        *,
        min_confidence_threshold: float = 0.32,
        min_term_coverage: float = 0.22,
        min_dense_similarity: float = 0.16,
        max_answer_sentences: int = 4,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.min_term_coverage = min_term_coverage
        self.min_dense_similarity = min_dense_similarity
        self.max_answer_sentences = max_answer_sentences

    def evaluate_confidence(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        threshold_override: float | None = None,
    ) -> AbstentionDecision:
        """Decide whether retrieved evidence is sufficient or the engine must abstain."""
        threshold = (
            threshold_override
            if threshold_override is not None
            else self.min_confidence_threshold
        )

        if not hits:
            return AbstentionDecision(
                should_abstain=True,
                confidence_score=0.0,
                reason="No matching chunks were retrieved from the indexed knowledge base.",
            )

        top_window = hits[: min(3, len(hits))]
        top_hit = top_window[0]

        best_coverage = max(h.signals.term_coverage for h in top_window)
        best_dense = max(h.signals.dense_sim for h in top_window)
        top_coverage = top_hit.signals.term_coverage
        top_dense = top_hit.signals.dense_sim
        top_header = top_hit.signals.title_section_match
        top_phrase = top_hit.signals.phrase_boost

        # Mode-invariant evidence confidence score bounded to [0, 1]
        confidence = (
            0.40 * (0.7 * top_coverage + 0.3 * best_coverage)
            + 0.36 * (0.7 * top_dense + 0.3 * best_dense)
            + 0.14 * top_header
            + 0.10 * top_phrase
        )
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        # Require at least some lexical OR strong semantic support
        q_tokens = content_tokens(query)
        has_zero_lexical = len(q_tokens) > 0 and best_coverage < 0.08
        weak_dual_signal = (
            best_coverage < self.min_term_coverage and best_dense < self.min_dense_similarity
        )

        if confidence < threshold or weak_dual_signal or (has_zero_lexical and best_dense < 0.28):
            reason = (
                f"Evidence confidence ({confidence:.2f}) fell below threshold ({threshold:.2f}) "
                f"with term_coverage={best_coverage:.2f} and dense_sim={best_dense:.2f}."
            )
            return AbstentionDecision(
                should_abstain=True,
                confidence_score=confidence,
                reason=reason,
            )

        return AbstentionDecision(
            should_abstain=False,
            confidence_score=confidence,
            reason=None,
        )

    @staticmethod
    def build_abstention_message(
        *,
        language: Literal["id", "en"],
        confidence: float,
        threshold: float,
    ) -> str:
        """Produce a transparent, non-hallucinatory abstention response in ID or EN."""
        if language == "id":
            return (
                "Maaf, saya tidak dapat menjawab pertanyaan ini secara akurat karena tidak ditemukan "
                "bukti yang cukup relevan di dalam basis pengetahuan yang terindeks "
                f"(skor kepercayaan bukti {confidence:.2f} di bawah ambang batas {threshold:.2f}). "
                "Silakan unggah dokumen terkait atau perjelas pertanyaan Anda agar sistem dapat "
                "memberikan jawaban bersumber sitasi tanpa berspekulasi."
            )
        return (
            "I cannot answer this question reliably because the indexed knowledge base does not "
            "contain sufficiently relevant supporting evidence "
            f"(evidence confidence {confidence:.2f} is below the required threshold {threshold:.2f}). "
            "To prevent hallucination, I have abstained rather than speculating. Please ingest the "
            "relevant source document or refine your query."
        )

    def synthesize_extractive_answer(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        language: Literal["id", "en"] = "en",
    ) -> str:
        """Synthesize a concise, strictly grounded answer with inline [doc_id#chunk_id] citations."""
        if not hits:
            return self.build_abstention_message(
                language=language,
                confidence=0.0,
                threshold=self.min_confidence_threshold,
            )

        q_tokens = content_tokens(query)
        q_set = set(q_tokens)
        q_bigrams = extract_bigrams(q_tokens)
        q_char = char_ngrams(query, n=4)
        prefers_numeric = bool(_NUMERIC_QUERY_RE.search(query))

        candidates: list[tuple[float, int, str, RetrievalHit]] = []
        top_hits = hits[: min(4, len(hits))]

        for rank_idx, hit in enumerate(top_hits):
            sentences = split_sentences(hit.chunk.text)
            if not sentences:
                sentences = [hit.chunk.text.strip()]

            rank_prior = 1.0 / (1.0 + 0.35 * rank_idx)
            for sent in sentences:
                clean_sent = re.sub(r"\s+", " ", sent).strip()
                if len(clean_sent) < 18:
                    continue
                s_tokens = content_tokens(clean_sent)
                if not s_tokens:
                    continue
                s_set = set(s_tokens)

                # Lexical recall of query concepts in this sentence
                q_overlap = len(q_set & s_set) / max(len(q_set), 1)
                s_bigrams = extract_bigrams(s_tokens)
                bg_overlap = (len(q_bigrams & s_bigrams) / len(q_bigrams)) if q_bigrams else 0.0
                s_char = char_ngrams(clean_sent, n=4)
                c_overlap = (len(q_char & s_char) / max(len(q_char), 1)) if q_char else 0.0

                num_bonus = 0.10 if (prefers_numeric and _HAS_NUMBER_RE.search(clean_sent)) else 0.0
                # Mild preference for informative complete sentences (40..320 chars)
                len_bonus = 0.05 if 45 <= len(clean_sent) <= 320 else 0.0

                sent_score = (
                    0.46 * q_overlap
                    + 0.22 * c_overlap
                    + 0.14 * bg_overlap
                    + 0.18 * rank_prior * max(hit.signals.dense_sim, hit.rerank_score)
                    + num_bonus
                    + len_bonus
                )
                candidates.append((sent_score, rank_idx, clean_sent, hit))

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[str, RetrievalHit]] = []
        selected_token_sets: list[set[str]] = []

        for score, _rank_idx, sent, hit in candidates:
            if len(selected) >= self.max_answer_sentences:
                break
            if selected and score < 0.16:
                break
            s_set = set(content_tokens(sent))
            # Avoid near-duplicate sentences across overlapping chunks
            is_dup = any(
                (len(s_set & prev) / max(len(s_set | prev), 1)) > 0.65
                for prev in selected_token_sets
            )
            if is_dup:
                continue
            selected.append((sent, hit))
            selected_token_sets.append(s_set)

        if not selected:
            first_hit = top_hits[0]
            fallback_snippet = re.sub(r"\s+", " ", first_hit.chunk.text).strip()[:280]
            selected = [(fallback_snippet, first_hit)]

        lines: list[str] = []
        for sent, hit in selected:
            clean = sent.rstrip()
            if clean[-1] not in ".!?":
                clean += "."
            # Strip existing bracketed citations if any to avoid nesting
            clean = _CITATION_TAG_RE.sub("", clean).strip()
            lines.append(f"{clean} [{hit.chunk.citation_key}]")

        return " ".join(lines)

    def verify_answer(
        self,
        answer: str,
        hits: Sequence[RetrievalHit],
    ) -> VerificationResult:
        """Verify every claim sentence against its cited chunk(s) to compute faithfulness_score."""
        hit_map = {h.chunk.citation_key: h for h in hits}
        # Split answer into sentences while keeping trailing citation tags attached
        raw_claims = self._split_answer_claims(answer)
        if not raw_claims:
            return VerificationResult(
                faithfulness_score=0.0,
                supported_claims=0,
                total_claims=0,
                citations=[],
            )

        claim_scores: list[float] = []
        supported_claims = 0
        citation_best_support: dict[str, float] = {}

        for claim in raw_claims:
            cited_keys = _CITATION_TAG_RE.findall(claim)
            valid_keys = [k for k in cited_keys if k in hit_map]
            claim_body = _CITATION_TAG_RE.sub("", claim).strip()
            claim_tokens = content_tokens(claim_body)

            if not claim_tokens:
                continue

            if not valid_keys:
                claim_scores.append(0.0)
                continue

            # Combine text from all chunks cited in this sentence
            combined_source = " ".join(
                f"{hit_map[k].chunk.title} {hit_map[k].chunk.section} {hit_map[k].chunk.text}"
                for k in valid_keys
            )
            source_tokens = set(content_tokens(combined_source))
            matched = sum(1 for t in claim_tokens if t in source_tokens)
            token_precision = matched / len(claim_tokens)

            claim_ngrams = char_ngrams(claim_body, n=4)
            source_ngrams = char_ngrams(combined_source, n=4)
            char_containment = (
                len(claim_ngrams & source_ngrams) / len(claim_ngrams) if claim_ngrams else 0.0
            )

            support = round(min(1.0, 0.75 * token_precision + 0.25 * char_containment), 4)
            claim_scores.append(support)
            if support >= 0.55:
                supported_claims += 1

            for key in valid_keys:
                prev = citation_best_support.get(key, 0.0)
                if support > prev:
                    citation_best_support[key] = support

        total_claims = len(claim_scores)
        faithfulness = (
            round(sum(claim_scores) / total_claims, 4) if total_claims > 0 else 0.0
        )

        citations: list[Citation] = []
        for key, score in citation_best_support.items():
            chunk = hit_map[key].chunk
            snippet = re.sub(r"\s+", " ", chunk.text).strip()
            if len(snippet) > 220:
                snippet = snippet[:217].rstrip() + "..."
            citations.append(
                Citation(
                    citation_key=key,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    section=chunk.section,
                    page=chunk.page,
                    snippet=snippet,
                    support_score=round(score, 4),
                )
            )

        return VerificationResult(
            faithfulness_score=faithfulness,
            supported_claims=supported_claims,
            total_claims=total_claims,
            citations=citations,
        )

    @staticmethod
    def _split_answer_claims(answer: str) -> list[str]:
        """Split an answer with inline [doc#chunk] citations into claim units."""
        text = normalize_text(answer).strip()
        if not text:
            return []
        # Split after citation tag followed by space or newline, or standard sentence end
        parts = re.split(r"(?<=\])\s+(?=[A-Z0-9\-\*\"'])|\n+", text)
        claims: list[str] = []
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) >= 12:
                claims.append(cleaned)
        return claims or [text]
