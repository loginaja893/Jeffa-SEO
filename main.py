# JeffaSEO — AI SEO toolkit (single-file). SERP-oriented keyword analysis, meta generation, sitemaps, scoring. Use for agents and on-chain claim payloads.
# No dependencies beyond stdlib; optional: requests for fetch. Populated defaults; no placeholders.

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator
from xml.etree import ElementTree as ET


# ------------------------------------------------------------------------------
# Constants (unique to JeffaSEO; do not reuse in other projects)
# ------------------------------------------------------------------------------

JEFFA_NAMESPACE = "jeffa_seo_v1"
JEFFA_DEFAULT_LOCALE = "en_US"
JEFFA_MAX_TITLE_LEN = 60
JEFFA_MAX_DESC_LEN = 160
JEFFA_MIN_DESC_LEN = 120
JEFFA_KEYWORD_DENSITY_FLOOR = 0.005
JEFFA_KEYWORD_DENSITY_CEIL = 0.03
JEFFA_SITEMAP_MAX_URLS = 50000
JEFFA_SITEMAP_INDEX_MAX = 50000
JEFFA_SCORE_BPS_CAP = 10000
JEFFA_STOP_WORDS_EN = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "been", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought"
})
JEFFA_META_VIEWPORT = "width=device-width, initial-scale=1.0"
JEFFA_DEFAULT_CHARSET = "UTF-8"
JEFFA_SCHEMA_ORG_CONTEXT = "https://schema.org"
JEFFA_CRAWL_DEFAULT_DELAY_MS = 1200
JEFFA_SERP_SNIPPET_MAX_TITLE = 60
JEFFA_SERP_SNIPPET_MAX_DESC = 155
JEFFA_READABILITY_MIN_WORDS = 300
JEFFA_READABILITY_IDEAL_WORDS = 800
JEFFA_H1_MAX_COUNT_RECOMMEND = 1
JEFFA_HASH_ALGO = "sha256"
JEFFA_ANCHOR_SEED = "jeffa_anchor_seed_7f3b9e2a"


class SerpTier(Enum):
    CORE = 1
    LONG_TAIL = 2
    BRAND = 3
    LOCAL = 4
    IMAGE = 5


class ContentGrade(Enum):
    A = 90
    B = 75
    C = 60
    D = 40
    F = 0


# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------

@dataclass
class KeywordResult:
    keyword: str
    count: int
    density_bps: int
    position_first: int
    position_last: int
    tier: SerpTier
    normalized: str


@dataclass
class MetaTags:
    title: str
    description: str
    canonical: str
    og_title: str
    og_description: str
    og_type: str
    twitter_card: str
    twitter_title: str
    twitter_description: str
    robots: str
    viewport: str
    charset: str
    locale: str
    extra: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SerpSnippet:
    title: str
    url: str
    display_url: str
    description: str
    breadcrumb: str
    score_bps: int


@dataclass
class PageScore:
    total_bps: int
    title_score_bps: int
    desc_score_bps: int
    h1_score_bps: int
    keyword_score_bps: int
    length_score_bps: int
    grade: ContentGrade
    suggestions: list[str]


@dataclass
class SitemapUrl:
    loc: str
    lastmod: str | None
    changefreq: str | None
    priority: float | None


# ------------------------------------------------------------------------------
# Text normalization and hashing (for claim ids / keyword hashes)
# ------------------------------------------------------------------------------

def jeffa_normalize_keyword(raw: str) -> str:
    s = raw.strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def jeffa_keyword_hash(keyword: str) -> str:
    norm = jeffa_normalize_keyword(keyword)
    payload = f"{JEFFA_NAMESPACE}:{norm}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jeffa_claim_id(agent_id: str, keyword: str, nonce: str) -> str:
    norm_kw = jeffa_normalize_keyword(keyword)
    payload = f"{JEFFA_ANCHOR_SEED}:{agent_id}:{norm_kw}:{nonce}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jeffa_agent_id(wallet_or_name: str) -> str:
    return hashlib.sha256(f"{JEFFA_NAMESPACE}:agent:{wallet_or_name}".encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------
# Keyword extraction and density
# ------------------------------------------------------------------------------

def jeffa_tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return [t.lower() for t in text.split() if t]


def jeffa_extract_keywords(text: str, min_len: int = 2, stop_words: frozenset[str] | None = None) -> list[str]:
    stop = stop_words or JEFFA_STOP_WORDS_EN
    tokens = jeffa_tokenize(text)
    return [t for t in tokens if len(t) >= min_len and t not in stop]


def jeffa_keyword_density_bps(text: str, keyword: str) -> int:
    tokens = jeffa_tokenize(text)
    if not tokens:
        return 0
    kw_norm = jeffa_normalize_keyword(keyword)
    kw_tokens = jeffa_tokenize(kw_norm)
    if not kw_tokens:
        return 0
    count = 0
    for i in range(len(tokens) - len(kw_tokens) + 1):
        if tokens[i:i + len(kw_tokens)] == kw_tokens:
            count += 1
    return (count * 10000) // len(tokens) if tokens else 0


def jeffa_analyze_keyword_in_text(text: str, keyword: str) -> KeywordResult:
    norm_kw = jeffa_normalize_keyword(keyword)
    tokens = jeffa_tokenize(text)
    kw_tokens = jeffa_tokenize(norm_kw)
    count = 0
    first = -1
    last = -1
    for i in range(len(tokens) - len(kw_tokens) + 1):
        if tokens[i:i + len(kw_tokens)] == kw_tokens:
            count += 1
            if first < 0:
                first = i
            last = i
    density_bps = (count * 10000) // len(tokens) if tokens else 0
    tier = _jeffa_infer_tier(keyword, count)
    return KeywordResult(
        keyword=keyword,
        count=count,
        density_bps=density_bps,
        position_first=first,
        position_last=last,
        tier=tier,
        normalized=norm_kw,
    )


def _jeffa_infer_tier(keyword: str, count: int) -> SerpTier:
    word_count = len(keyword.split())
    if word_count >= 4:
        return SerpTier.LONG_TAIL
    if re.search(r"\b(near me|city|zip|address)\b", keyword, re.I):
        return SerpTier.LOCAL
    if re.search(r"\b(image|photo|picture|png|jpg)\b", keyword, re.I):
        return SerpTier.IMAGE
    if word_count <= 2 and count >= 3:
        return SerpTier.BRAND
    return SerpTier.CORE


def jeffa_top_ngrams(text: str, n: int = 2, top_k: int = 20) -> list[tuple[str, int]]:
    tokens = jeffa_extract_keywords(text)
    if len(tokens) < n:
        return []
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams).most_common(top_k)


# ------------------------------------------------------------------------------
# Meta tag generation
# ------------------------------------------------------------------------------

def jeffa_truncate_for_meta(s: str, max_len: int, suffix: str = "") -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)].rsplit(" ", 1)[0] + suffix


def jeffa_build_meta(
    title: str,
    description: str,
    canonical: str = "",
    og_type: str = "website",
    twitter_card: str = "summary_large_image",
    robots: str = "index, follow",
    locale: str = JEFFA_DEFAULT_LOCALE,
    extra: list[tuple[str, str]] | None = None,
) -> MetaTags:
    title_trim = jeffa_truncate_for_meta(title, JEFFA_MAX_TITLE_LEN)
    desc_trim = description.strip()
    if len(desc_trim) > JEFFA_MAX_DESC_LEN:
        desc_trim = jeffa_truncate_for_meta(desc_trim, JEFFA_MAX_DESC_LEN, "...")
    if len(desc_trim) < JEFFA_MIN_DESC_LEN and len(description) >= JEFFA_MIN_DESC_LEN:
        desc_trim = jeffa_truncate_for_meta(description, JEFFA_MAX_DESC_LEN, "...")
    return MetaTags(
        title=title_trim,
        description=desc_trim,
        canonical=canonical,
        og_title=title_trim,
        og_description=desc_trim,
        og_type=og_type,
        twitter_card=twitter_card,
        twitter_title=title_trim,
        twitter_description=desc_trim,
        robots=robots,
        viewport=JEFFA_META_VIEWPORT,
        charset=JEFFA_DEFAULT_CHARSET,
        locale=locale,
        extra=extra or [],
    )


def jeffa_meta_to_html(meta: MetaTags, indent: str = "  ") -> str:
    lines = [
        f'<meta charset="{html.escape(meta.charset)}">',
        f'<meta name="viewport" content="{html.escape(meta.viewport)}">',
        f'<title>{html.escape(meta.title)}</title>',
        f'<meta name="description" content="{html.escape(meta.description)}">',
        f'<meta name="robots" content="{html.escape(meta.robots)}">',
    ]
    if meta.canonical:
        lines.append(f'<link rel="canonical" href="{html.escape(meta.canonical)}">')
    lines.extend([
        f'<meta property="og:title" content="{html.escape(meta.og_title)}">',
