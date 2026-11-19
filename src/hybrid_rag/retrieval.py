"""Hybrid retrieval engine: Okapi BM25, Dense Vector search, RRF, and Cross-Signal Reranking."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

import httpx
import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from hybrid_rag.config import Settings
from hybrid_rag.schemas import Chunk, RetrievalHit, RetrievalMode, RetrievalSignals
from hybrid_rag.text_utils import content_tokens, extract_bigrams, normalize_text, tokenize


class BM25Retriever:
    """Okapi BM25 retriever with bilingual ID/EN tokenization and structural field boost."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avgdl: float = 1.0
        self._idf: dict[str, float] = {}
        self._doc_count: int = 0

    def fit(self, chunks: Sequence[Chunk]) -> None:
        """Build inverted document frequency and term frequency tables."""
        self.chunks = list(chunks)
        self._doc_count = len(self.chunks)
        self._doc_term_freqs = []
        self._doc_lengths = []
        df: Counter[str] = Counter()

        for chunk in self.chunks:
            # Index exact + lightly stemmed tokens and lightly boost title/section tokens
            body_tokens = tokenize(chunk.text, remove_stopwords=True, apply_stem=False)
            body_stemmed = content_tokens(chunk.text)
            header_tokens = content_tokens(f"{chunk.title} {chunk.section}")

            tf: Counter[str] = Counter(body_tokens)
            # Add stemmed variants (without double-counting identical tokens)
            for tok in body_stemmed:
                if tok not in tf:
                    tf[tok] += 1
            for tok in header_tokens:
                tf[tok] += 1

            self._doc_term_freqs.append(tf)
            doc_len = sum(tf.values())
            self._doc_lengths.append(doc_len)
            for term in tf:
                df[term] += 1

        self._avgdl = (sum(self._doc_lengths) / self._doc_count) if self._doc_count else 1.0
        # Lucene-style positive BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self._idf = {
            term: math.log(1.0 + (self._doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def term_idf(self, term: str) -> float:
        """Return BM25 IDF weight for a term (defaults to high weight for OOV terms)."""
        if term in self._idf:
            return self._idf[term]
        return math.log(1.0 + (self._doc_count + 0.5) / 0.5) if self._doc_count else 1.0

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float, list[str]]]:
        """Return [(chunk_idx, bm25_score, matched_terms)] sorted descending."""
        if not self.chunks:
            return []

        raw_q = tokenize(query, remove_stopwords=True, apply_stem=False)
        stemmed_q = content_tokens(query)
        query_terms = list(dict.fromkeys(raw_q + stemmed_q))
        if not query_terms:
            return []

        scored: list[tuple[int, float, list[str]]] = []
        for idx, tf in enumerate(self._doc_term_freqs):
            dl = self._doc_lengths[idx]
            norm_factor = self.k1 * (1.0 - self.b + self.b * (dl / max(self._avgdl, 1e-6)))
            score = 0.0
            matched: list[str] = []
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * ((freq * (self.k1 + 1.0)) / (freq + norm_factor))
                matched.append(term)
            if score > 0.0:
                scored.append((idx, score, matched))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class EmbeddingBackend(ABC):
    """Pluggable interface for offline and remote dense embedding providers."""

    name: str = "base"

    @abstractmethod
    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Fit (if applicable) and return L2-normalized float32 embeddings of shape (N, D)."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into an L2-normalized float32 vector of shape (D,)."""


class LocalSVDEmbeddingBackend(EmbeddingBackend):
    """Offline dense semantic embedder using word + char_wb TF-IDF + TruncatedSVD (LSA)."""

    def __init__(self, n_components: int = 128, random_state: int = 42) -> None:
        self.name = f"local-svd-{n_components}"
        self.n_components = n_components
        self.random_state = random_state
        self._word_vec = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9._/-]*\b",
            max_features=8000,
        )
        self._char_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            sublinear_tf=True,
            ngram_range=(3, 5),
            max_features=12000,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def _vectorize(self, texts: Sequence[str], *, fit: bool) -> np.ndarray | object:
        cleaned = [normalize_text(t) for t in texts]
        if fit:
            w_mat = self._word_vec.fit_transform(cleaned)
            c_mat = self._char_vec.fit_transform(cleaned)
        else:
            w_mat = self._word_vec.transform(cleaned)
            c_mat = self._char_vec.transform(cleaned)
        # Weight word features slightly higher while retaining subword char_wb morphology
        return hstack([w_mat * 1.15, c_mat * 0.85]).tocsr()

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            self._fitted = False
            return np.zeros((0, 1), dtype=np.float32)

        sparse_mat = self._vectorize(texts, fit=True)
        n_samples, n_features = sparse_mat.shape
        max_rank = min(self.n_components, n_samples - 1, n_features - 1)

        if max_rank >= 2:
            self._svd = TruncatedSVD(n_components=max_rank, random_state=self.random_state)
            dense = self._svd.fit_transform(sparse_mat)
        else:
            self._svd = None
            dense = sparse_mat.toarray()

        dense_norm = normalize(dense, norm="l2", axis=1).astype(np.float32)
        self._fitted = True
        return dense_norm

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            return np.zeros((1,), dtype=np.float32)
        sparse_q = self._vectorize([query], fit=False)
        dense_q = self._svd.transform(sparse_q) if self._svd is not None else sparse_q.toarray()
        return normalize(dense_q, norm="l2", axis=1)[0].astype(np.float32)


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Remote dense embeddings via Ollama `/api/embeddings`."""

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _fetch(self, text: str) -> np.ndarray:
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack([self._fetch(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._fetch(query)


class GeminiEmbeddingBackend(EmbeddingBackend):
    """Remote dense embeddings via Google Gemini `embedContent` REST API."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.name = f"gemini:{model}"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _fetch(self, text: str) -> np.ndarray:
        url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"
        resp = httpx.post(
            url,
            json={"content": {"parts": [{"text": text}]}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        vec = np.asarray(resp.json()["embedding"]["values"], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack([self._fetch(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._fetch(query)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Remote dense embeddings via OpenAI `/v1/embeddings` REST API."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.name = f"openai:{model}"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _fetch_batch(self, texts: Sequence[str]) -> np.ndarray:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda item: item["index"])
        mat = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        return normalize(mat, norm="l2", axis=1).astype(np.float32)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self._fetch_batch(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._fetch_batch([query])[0]


def create_embedding_backend(settings: Settings) -> EmbeddingBackend:
    """Instantiate configured embedding backend, falling back safely to local SVD."""
    provider = settings.embedding_provider
    if provider == "ollama":
        return OllamaEmbeddingBackend(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            timeout=settings.llm_timeout_seconds,
        )
    if provider == "gemini" and settings.gemini_api_key:
        return GeminiEmbeddingBackend(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            model=settings.embedding_model,
            timeout=settings.llm_timeout_seconds,
        )
    if provider == "openai" and settings.openai_api_key:
        return OpenAIEmbeddingBackend(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            timeout=settings.llm_timeout_seconds,
        )
    return LocalSVDEmbeddingBackend(n_components=settings.embedding_dim)


class DenseVectorRetriever:
    """L2-normalized dense vector retriever with cosine similarity search."""

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self.backend = backend or LocalSVDEmbeddingBackend()
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray = np.zeros((0, 1), dtype=np.float32)

    @staticmethod
    def _chunk_repr(chunk: Chunk) -> str:
        return f"{chunk.title}\n{chunk.section}\n{chunk.text}"

    def fit(self, chunks: Sequence[Chunk]) -> np.ndarray:
        """Compute L2-normalized embeddings for all chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return self.embeddings
        texts = [self._chunk_repr(c) for c in self.chunks]
        self.embeddings = self.backend.fit_transform(texts)
        return self.embeddings

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(chunk_idx, cosine_similarity)] sorted descending."""
        if not self.chunks or self.embeddings.shape[0] == 0:
            return []
        q_vec = self.backend.embed_query(query)
        if q_vec.shape[0] != self.embeddings.shape[1]:
            return []
        sims = self.embeddings @ q_vec
        # Clip to [0, 1] since negative cosine angles indicate orthogonal/unrelated topics
        sims = np.clip(sims, 0.0, 1.0)
        top_indices = np.argsort(-sims)[:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def score_all(self, query: str) -> np.ndarray:
        """Compute clipped cosine similarity for every chunk."""
        if not self.chunks or self.embeddings.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        q_vec = self.backend.embed_query(query)
        if q_vec.shape[0] != self.embeddings.shape[1]:
            return np.zeros((len(self.chunks),), dtype=np.float32)
        return np.clip(self.embeddings @ q_vec, 0.0, 1.0)


class ReciprocalRankFusion:
    """Rank-based fusion combining BM25 and Dense ranked lists via 1 / (k + rank)."""

    def __init__(self, k: int = 60, bm25_weight: float = 1.0, dense_weight: float = 1.0) -> None:
        self.k = k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def fuse(
        self,
        bm25_ranked: Sequence[tuple[int, float, list[str]]],
        dense_ranked: Sequence[tuple[int, float]],
    ) -> list[dict[str, object]]:
        """Merge ranked lists while preserving per-retriever ranks and raw scores."""
        merged: dict[int, dict[str, object]] = {}

        for rank, (idx, score, matched) in enumerate(bm25_ranked, start=1):
            entry = merged.setdefault(
                idx,
                {
                    "idx": idx,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "rrf_score": 0.0,
                    "matched_terms": [],
                },
            )
            entry["bm25_rank"] = rank
            entry["bm25_score"] = float(score)
            entry["matched_terms"] = list(matched)
            entry["rrf_score"] = float(entry["rrf_score"]) + self.bm25_weight / (self.k + rank)

        for rank, (idx, score) in enumerate(dense_ranked, start=1):
            entry = merged.setdefault(
                idx,
                {
                    "idx": idx,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "rrf_score": 0.0,
                    "matched_terms": [],
                },
            )
            entry["dense_rank"] = rank
            entry["dense_score"] = float(score)
            entry["rrf_score"] = float(entry["rrf_score"]) + self.dense_weight / (self.k + rank)

        results = list(merged.values())
        results.sort(
            key=lambda item: (float(item["rrf_score"]), float(item["dense_score"]), float(item["bm25_score"])),
            reverse=True,
        )
        return results


class CrossSignalReranker:
    """Second-stage reranker combining RRF, IDF term coverage, phrases, headers, and proximity."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self._max_rrf = 2.0 / (rrf_k + 1.0)

    def compute_signals(
        self,
        *,
        query: str,
        chunk: Chunk,
        rrf_score: float,
        dense_score: float,
        bm25_retriever: BM25Retriever,
    ) -> tuple[float, RetrievalSignals, list[str]]:
        """Compute bounded [0, 1] composite rerank score and explainable signal breakdown."""
        q_tokens = content_tokens(query)
        unique_q = list(dict.fromkeys(q_tokens))
        chunk_tokens = content_tokens(chunk.text)
        header_tokens = content_tokens(f"{chunk.title} {chunk.section}")
        combined_tokens = header_tokens + chunk_tokens
        combined_set = set(combined_tokens)
        header_set = set(header_tokens)

        # 1. Normalized RRF consensus score
        rrf_norm = min(1.0, rrf_score / self._max_rrf) if self._max_rrf > 0 else 0.0

        # 2. IDF-weighted query term coverage
        if unique_q:
            total_idf = sum(bm25_retriever.term_idf(t) for t in unique_q)
            matched_terms = [t for t in unique_q if t in combined_set]
            matched_idf = sum(bm25_retriever.term_idf(t) for t in matched_terms)
            term_coverage = matched_idf / max(total_idf, 1e-6)
        else:
            matched_terms = []
            term_coverage = 0.0

        # 3. Exact phrase & bigram match boost
        norm_q = normalize_text(query).lower().strip()
        norm_full = normalize_text(f"{chunk.title} {chunk.section} {chunk.text}").lower()
        q_bigrams = extract_bigrams(q_tokens)
        c_bigrams = extract_bigrams(combined_tokens)
        bigram_overlap = (len(q_bigrams & c_bigrams) / len(q_bigrams)) if q_bigrams else 0.0
        exact_substring = 1.0 if (len(norm_q) >= 8 and norm_q in norm_full) else 0.0
        phrase_boost = min(1.0, 0.7 * bigram_overlap + 0.3 * exact_substring)

        # 4. Title & section heading alignment
        if unique_q:
            header_idf = sum(bm25_retriever.term_idf(t) for t in unique_q if t in header_set)
            total_idf = sum(bm25_retriever.term_idf(t) for t in unique_q)
            title_section_match = min(1.0, header_idf / max(total_idf, 1e-6))
        else:
            title_section_match = 0.0

        # 5. Term proximity: tighter span between distinct matched query terms in chunk
        proximity = self._compute_proximity(matched_terms, chunk_tokens)

        # 6. Dense cosine similarity
        dense_sim = max(0.0, min(1.0, float(dense_score)))

        # Weighted linear combination bounded to [0, 1]
        rerank_score = (
            0.32 * rrf_norm
            + 0.25 * term_coverage
            + 0.18 * dense_sim
            + 0.10 * phrase_boost
            + 0.08 * title_section_match
            + 0.07 * proximity
        )
        rerank_score = round(max(0.0, min(1.0, rerank_score)), 4)

        signals = RetrievalSignals(
            rrf_norm=round(rrf_norm, 4),
            term_coverage=round(term_coverage, 4),
            phrase_boost=round(phrase_boost, 4),
            title_section_match=round(title_section_match, 4),
            proximity=round(proximity, 4),
            dense_sim=round(dense_sim, 4),
        )
        return rerank_score, signals, matched_terms

    @staticmethod
    def _compute_proximity(matched_terms: Sequence[str], chunk_tokens: Sequence[str]) -> float:
        """Score how tightly matched query terms cluster within the chunk text."""
        if len(matched_terms) <= 1:
            return 0.5 if len(matched_terms) == 1 else 0.0

        positions: dict[str, list[int]] = {}
        target_set = set(matched_terms)
        for pos, tok in enumerate(chunk_tokens):
            if tok in target_set:
                positions.setdefault(tok, []).append(pos)

        present_terms = list(positions.keys())
        if len(present_terms) <= 1:
            return 0.35 if present_terms else 0.0

        # Sliding window minimum span covering all present matched terms
        merged_pos = sorted(
            (pos, term) for term, pos_list in positions.items() for pos in pos_list
        )
        counts: Counter[str] = Counter()
        needed = len(present_terms)
        have = 0
        left = 0
        best_span = len(chunk_tokens) + 1

        for right, (r_pos, r_term) in enumerate(merged_pos):
            counts[r_term] += 1
            if counts[r_term] == 1:
                have += 1
            while have == needed and left <= right:
                l_pos, l_term = merged_pos[left]
                span = r_pos - l_pos + 1
                if span < best_span:
                    best_span = span
                counts[l_term] -= 1
                if counts[l_term] == 0:
                    have -= 1
                left += 1

        if best_span > len(chunk_tokens):
            return 0.0
        # Ideal tight span == needed -> 1.0; decays smoothly as distance widens
        excess = max(0, best_span - needed)
        return float(1.0 / (1.0 + excess / 12.0))


class HybridRetriever:
    """Unified retriever supporting 'bm25', 'dense', and 'hybrid' (RRF + Rerank) modes."""

    def __init__(
        self,
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        candidate_pool_size: int = 25,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.dense = DenseVectorRetriever(backend=embedding_backend)
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.reranker = CrossSignalReranker(rrf_k=rrf_k)
        self.candidate_pool_size = candidate_pool_size
        self.chunks: list[Chunk] = []

    def index(self, chunks: Sequence[Chunk]) -> np.ndarray:
        """Fit both BM25 and Dense indices over all chunks."""
        self.chunks = list(chunks)
        self.bm25.fit(self.chunks)
        return self.dense.fit(self.chunks)

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Retrieve and score top-k chunks in bm25, dense, or hybrid mode."""
        if not self.chunks:
            return []

        pool = max(self.candidate_pool_size, top_k * 3)
        bm25_ranked = self.bm25.search(query, top_k=pool)
        dense_ranked = self.dense.search(query, top_k=pool)
        all_dense_scores = self.dense.score_all(query)

        bm25_map: dict[int, tuple[int, float, list[str]]] = {
            idx: (rank, score, matched)
            for rank, (idx, score, matched) in enumerate(bm25_ranked, start=1)
        }
        dense_map: dict[int, tuple[int, float]] = {
            idx: (rank, score) for rank, (idx, score) in enumerate(dense_ranked, start=1)
        }

        if mode == "bm25":
            hits: list[RetrievalHit] = []
            for rank, (idx, b_score, matched) in enumerate(bm25_ranked[:top_k], start=1):
                chunk = self.chunks[idx]
                d_rank, d_score = dense_map.get(idx, (None, float(all_dense_scores[idx])))
                rrf_val = 1.0 / (self.rrf.k + rank)
                rerank_score, signals, sig_terms = self.reranker.compute_signals(
                    query=query,
                    chunk=chunk,
                    rrf_score=rrf_val,
                    dense_score=d_score,
                    bm25_retriever=self.bm25,
                )
                hits.append(
                    RetrievalHit(
                        chunk=chunk,
                        rank=rank,
                        score=round(b_score, 4),
                        bm25_rank=rank,
                        bm25_score=round(b_score, 4),
                        dense_rank=d_rank,
                        dense_score=round(d_score, 4),
                        rrf_score=round(rrf_val, 5),
                        rerank_score=rerank_score,
                        matched_terms=sorted(set(matched) | set(sig_terms)),
                        signals=signals,
                    )
                )
            return hits

        if mode == "dense":
            hits = []
            for rank, (idx, d_score) in enumerate(dense_ranked[:top_k], start=1):
                chunk = self.chunks[idx]
                b_entry = bm25_map.get(idx)
                b_rank = b_entry[0] if b_entry else None
                b_score = b_entry[1] if b_entry else 0.0
                matched = b_entry[2] if b_entry else []
                rrf_val = 1.0 / (self.rrf.k + rank)
                rerank_score, signals, sig_terms = self.reranker.compute_signals(
                    query=query,
                    chunk=chunk,
                    rrf_score=rrf_val,
                    dense_score=d_score,
                    bm25_retriever=self.bm25,
                )
                hits.append(
                    RetrievalHit(
                        chunk=chunk,
                        rank=rank,
                        score=round(d_score, 4),
                        bm25_rank=b_rank,
                        bm25_score=round(b_score, 4),
                        dense_rank=rank,
                        dense_score=round(d_score, 4),
                        rrf_score=round(rrf_val, 5),
                        rerank_score=rerank_score,
                        matched_terms=sorted(set(matched) | set(sig_terms)),
                        signals=signals,
                    )
                )
            return hits

        # Hybrid mode: RRF fusion + Cross-Signal Reranking
        fused = self.rrf.fuse(bm25_ranked, dense_ranked)[:pool]
        scored_hits: list[RetrievalHit] = []
        for item in fused:
            idx = int(item["idx"])  # type: ignore[arg-type]
            chunk = self.chunks[idx]
            d_score = float(item["dense_score"])  # type: ignore[arg-type]
            if d_score == 0.0 and idx < len(all_dense_scores):
                d_score = float(all_dense_scores[idx])
            rrf_val = float(item["rrf_score"])  # type: ignore[arg-type]
            rerank_score, signals, sig_terms = self.reranker.compute_signals(
                query=query,
                chunk=chunk,
                rrf_score=rrf_val,
                dense_score=d_score,
                bm25_retriever=self.bm25,
            )
            matched_bm25 = list(item["matched_terms"])  # type: ignore[arg-type]
            scored_hits.append(
                RetrievalHit(
                    chunk=chunk,
                    rank=0,
                    score=rerank_score,
                    bm25_rank=item["bm25_rank"],  # type: ignore[arg-type]
                    bm25_score=round(float(item["bm25_score"]), 4),  # type: ignore[arg-type]
                    dense_rank=item["dense_rank"],  # type: ignore[arg-type]
                    dense_score=round(d_score, 4),
                    rrf_score=round(rrf_val, 5),
                    rerank_score=rerank_score,
                    matched_terms=sorted(set(matched_bm25) | set(sig_terms)),
                    signals=signals,
                )
            )

        scored_hits.sort(
            key=lambda h: (h.rerank_score, h.rrf_score, h.dense_score),
            reverse=True,
        )
        top_hits = scored_hits[:top_k]
        for rank, hit in enumerate(top_hits, start=1):
            hit.rank = rank
        return top_hits
