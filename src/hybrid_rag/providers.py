"""Pluggable async answer generation providers: offline extractive-grounded, Ollama, Gemini, OpenAI."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Literal

import httpx

from hybrid_rag.config import Settings
from hybrid_rag.grounding import GroundingEngine
from hybrid_rag.schemas import RetrievalHit

SYSTEM_GROUNDING_PROMPT = (
    "You are a strict enterprise RAG assistant. Answer ONLY using the provided context chunks. "
    "Every factual sentence MUST include an inline citation in the exact format [doc_id#chunk_id]. "
    "Respond in the same language as the user's question (Indonesian or English). "
    "If the provided context does not contain enough evidence to answer, explicitly abstain "
    "instead of guessing or relying on outside knowledge."
)


def build_context_block(hits: Sequence[RetrievalHit]) -> str:
    """Format retrieved chunks with explicit citation keys for LLM prompts."""
    blocks: list[str] = []
    for hit in hits:
        c = hit.chunk
        page_str = f" | Page {c.page}" if c.page is not None else ""
        blocks.append(
            f"[{c.citation_key}] ({c.title} — {c.section}{page_str})\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


class BaseAnswerProvider(ABC):
    """Abstract async answer synthesis adapter."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> str:
        """Generate a complete grounded answer with inline citations."""

    async def stream(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> AsyncIterator[str]:
        """Stream answer deltas; defaults to chunked emission of generate()."""
        full_text = await self.generate(query=query, hits=hits, language=language)
        words = full_text.split(" ")
        for idx, word in enumerate(words):
            suffix = " " if idx < len(words) - 1 else ""
            yield word + suffix
            await asyncio.sleep(0)


class ExtractiveGroundedProvider(BaseAnswerProvider):
    """Deterministic, high-precision offline grounded answer synthesizer with strict citations."""

    name = "extractive-grounded"

    def __init__(self, grounding: GroundingEngine) -> None:
        self.grounding = grounding

    async def generate(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> str:
        return self.grounding.synthesize_extractive_answer(
            query=query,
            hits=hits,
            language=language,
        )

    async def stream(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> AsyncIterator[str]:
        full_text = await self.generate(query=query, hits=hits, language=language)
        tokens = full_text.split(" ")
        step = 3
        for idx in range(0, len(tokens), step):
            piece = " ".join(tokens[idx : idx + step])
            if idx + step < len(tokens):
                piece += " "
            yield piece
            await asyncio.sleep(0.005)


class OllamaProvider(BaseAnswerProvider):
    """Local LLM adapter using Ollama `/api/chat` with streaming support."""

    def __init__(self, *, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _payload(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        stream: bool,
    ) -> dict[str, object]:
        context = build_context_block(hits)
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": SYSTEM_GROUNDING_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer with inline citations:",
                },
            ],
            "options": {"temperature": 0.1},
        }

    async def generate(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(query, hits, stream=False),
            )
            resp.raise_for_status()
            return str(resp.json()["message"]["content"]).strip()

    async def stream(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client, client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._payload(query, hits, stream=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield str(delta)


class GeminiProvider(BaseAnswerProvider):
    """Google Gemini REST API adapter via httpx."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self.name = f"gemini:{model}"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> str:
        context = build_context_block(hits)
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_GROUNDING_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"Context:\n{context}\n\nQuestion: {query}\n"
                                "Answer with inline citations:"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(str(p.get("text", "")) for p in parts).strip()


class OpenAIProvider(BaseAnswerProvider):
    """OpenAI-compatible `/v1/chat/completions` adapter with SSE streaming support."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self.name = f"openai:{model}"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        stream: bool,
    ) -> dict[str, object]:
        context = build_context_block(hits)
        return {
            "model": self.model,
            "temperature": 0.1,
            "stream": stream,
            "messages": [
                {"role": "system", "content": SYSTEM_GROUNDING_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer with inline citations:",
                },
            ],
        }

    async def generate(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(query, hits, stream=False),
            )
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"]).strip()

    async def stream(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client, client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(query, hits, stream=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    yield str(delta)


def create_answer_provider(
    settings: Settings,
    grounding: GroundingEngine,
) -> BaseAnswerProvider:
    """Instantiate configured answer provider, defaulting cleanly to offline extractive-grounded."""
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    return ExtractiveGroundedProvider(grounding=grounding)
