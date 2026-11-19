"""Bilingual Indonesian/English text normalization, tokenization, and language utilities."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

# Curated bilingual stopwords that preserve technical acronyms and negation semantics.
STOPWORDS_EN: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "must",
        "of",
        "on",
        "or",
        "our",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

STOPWORDS_ID: frozenset[str] = frozenset(
    {
        "ada",
        "adalah",
        "agar",
        "akan",
        "antara",
        "apa",
        "apabila",
        "apakah",
        "atau",
        "bagaimana",
        "bagi",
        "bahwa",
        "berapa",
        "bisa",
        "dalam",
        "dan",
        "dapat",
        "dari",
        "dengan",
        "di",
        "harus",
        "ini",
        "itu",
        "jika",
        "juga",
        "kapan",
        "karena",
        "ke",
        "kepada",
        "kita",
        "lain",
        "lebih",
        "maka",
        "mana",
        "maupun",
        "melakukan",
        "mengapa",
        "menjadi",
        "merupakan",
        "oleh",
        "pada",
        "para",
        "saat",
        "saja",
        "sebagai",
        "sebuah",
        "secara",
        "sedang",
        "sehingga",
        "selain",
        "seluruh",
        "semua",
        "serta",
        "setiap",
        "siapa",
        "sudah",
        "telah",
        "tentang",
        "terhadap",
        "tersebut",
        "untuk",
        "wajib",
        "yaitu",
        "yang",
    }
)

ALL_STOPWORDS: frozenset[str] = STOPWORDS_EN | STOPWORDS_ID

# Indicators used for lightweight Indonesian vs English detection
_ID_MARKERS: frozenset[str] = frozenset(
    {
        "apa",
        "apakah",
        "bagaimana",
        "berapa",
        "siapa",
        "kapan",
        "mengapa",
        "kenapa",
        "yang",
        "dan",
        "untuk",
        "dengan",
        "pada",
        "dari",
        "atau",
        "dalam",
        "tidak",
        "wajib",
        "jika",
        "apabila",
        "tentang",
        "menurut",
        "jelaskan",
        "sebutkan",
        "kebijakan",
        "pelindungan",
        "perlindungan",
        "sanksi",
        "pribadi",
        "bahasa",
        "indonesia",
        "rahasia",
        "enkripsi",
        "rotasi",
        "kebocoran",
        "hak",
        "data",
        "maksimal",
        "minimal",
        "kunci",
    }
)

_EN_MARKERS: frozenset[str] = frozenset(
    {
        "what",
        "how",
        "when",
        "why",
        "where",
        "which",
        "who",
        "the",
        "is",
        "are",
        "does",
        "should",
        "must",
        "with",
        "from",
        "between",
        "explain",
        "describe",
        "threshold",
        "thresholds",
        "latency",
        "rollback",
        "governance",
        "retrieval",
        "chunking",
        "metrics",
        "canary",
        "incident",
    }
)

# Common colloquial Indonesian -> standard mapping for query normalization
_ID_SLANG_MAP: dict[str, str] = {
    "yg": "yang",
    "dgn": "dengan",
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "gak": "tidak",
    "udh": "sudah",
    "blm": "belum",
    "utk": "untuk",
    "krn": "karena",
    "dlm": "dalam",
    "tdk": "tidak",
    "perlindungan": "pelindungan",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-\*\(])")
_ABBREV_PROTECTOR = re.compile(
    r"\b(No|UU|Dr|Prof|Mr|Mrs|Ms|e\.g|i\.e|vs|etc|Inc|Ltd|St)\.", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Normalize unicode, whitespace, and punctuation noise while keeping domain symbols."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def lightweight_stem(token: str) -> str:
    """Conservative bilingual stemmer for lexical recall without destructive over-stemming."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token

    # Indonesian possessive & particle suffixes
    for suffix in ("nya", "kah", "lah"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break

    # Indonesian common derivational suffixes
    for suffix in ("kan", "an"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 5:
            token = token[: -len(suffix)]
            break

    # English regular inflectional suffixes
    for suffix in ("ings", "ing", "edly", "ies", "ied", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            if suffix in {"ies", "ied"}:
                return token[: -len(suffix)] + "y"
            if suffix == "s" and token.endswith("ss"):
                return token
            return token[: -len(suffix)]
    return token


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stem: bool = False,
    expand_compounds: bool = True,
) -> list[str]:
    """Tokenize bilingual ID/EN text preserving acronyms, numeric thresholds, and sub-tokens."""
    cleaned = normalize_text(text).lower()
    raw_tokens = _TOKEN_RE.findall(cleaned)
    tokens: list[str] = []

    for tok in raw_tokens:
        # Strip leading/trailing punctuation artifacts
        tok = tok.strip("._/-")
        if not tok:
            continue
        tok = _ID_SLANG_MAP.get(tok, tok)

        parts = [tok]
        if expand_compounds and any(sep in tok for sep in ("-", "_", "/")):
            subparts = [p for p in re.split(r"[-_/]+", tok) if p]
            if len(subparts) > 1:
                parts.extend(subparts)

        for part in parts:
            part = _ID_SLANG_MAP.get(part, part)
            if len(part) == 1 and not part.isdigit():
                continue
            if remove_stopwords and part in ALL_STOPWORDS:
                continue
            if apply_stem:
                part = lightweight_stem(part)
            tokens.append(part)

    return tokens


def content_tokens(text: str) -> list[str]:
    """Extract normalized content tokens (stopwords removed, lightly stemmed)."""
    return tokenize(text, remove_stopwords=True, apply_stem=True, expand_compounds=True)


def extract_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Return adjacent token bigrams."""
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def char_ngrams(text: str, n: int = 4) -> set[str]:
    """Extract character n-grams over normalized alphanumerics for morphology-resilient overlap."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", normalize_text(text).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def detect_language(text: str) -> Literal["id", "en"]:
    """Detect whether a query/text is primarily Indonesian ('id') or English ('en')."""
    raw = _TOKEN_RE.findall(normalize_text(text).lower())
    id_hits = sum(1 for t in raw if t in _ID_MARKERS)
    en_hits = sum(1 for t in raw if t in _EN_MARKERS)
    if id_hits > en_hits:
        return "id"
    if en_hits > id_hits:
        return "en"
    # Secondary heuristic: Indonesian affix patterns
    id_affix = sum(
        1
        for t in raw
        if len(t) >= 6 and (t.startswith(("meng", "meny", "peng", "perl", "pel")) or t.endswith("nya"))
    )
    return "id" if id_affix > 0 else "en"


def split_sentences(text: str) -> list[str]:
    """Split prose & markdown bullet lines into clean sentences without breaking abbreviations."""
    text = normalize_text(text).strip()
    if not text:
        return []

    # Protect abbreviations like "No. 27" or "e.g."
    protected = _ABBREV_PROTECTOR.sub(lambda m: m.group(1) + "<DOT>", text)
    sentences: list[str] = []

    for line in protected.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Remove leading markdown heading / bullet markers for cleaner sentence units
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        stripped = re.sub(r"^\d+\.\s+", "", stripped)
        if not stripped:
            continue
        parts = _SENTENCE_SPLIT_RE.split(stripped)
        for part in parts:
            restored = part.replace("<DOT>", ".").strip()
            if len(restored) >= 12:
                sentences.append(restored)

    return sentences
