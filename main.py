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
        f'<meta property="og:description" content="{html.escape(meta.og_description)}">',
        f'<meta property="og:type" content="{html.escape(meta.og_type)}">',
        f'<meta name="twitter:card" content="{html.escape(meta.twitter_card)}">',
        f'<meta name="twitter:title" content="{html.escape(meta.twitter_title)}">',
        f'<meta name="twitter:description" content="{html.escape(meta.twitter_description)}">',
    ])
    for name, content in meta.extra:
        lines.append(f'<meta name="{html.escape(name)}" content="{html.escape(content)}">')
    return "\n".join(indent + line for line in lines)


# ------------------------------------------------------------------------------
# SERP snippet generation and scoring
# ------------------------------------------------------------------------------

def jeffa_serp_snippet(
    title: str,
    url: str,
    description: str,
    breadcrumb: str = "",
    score_bps: int = 0,
) -> SerpSnippet:
    display_url = urllib.parse.urlparse(url).netloc or url
    title_cut = jeffa_truncate_for_meta(title, JEFFA_SERP_SNIPPET_MAX_TITLE, "...")
    desc_cut = jeffa_truncate_for_meta(description, JEFFA_SERP_SNIPPET_MAX_DESC, "...")
    return SerpSnippet(
        title=title_cut,
        url=url,
        display_url=display_url,
        description=desc_cut,
        breadcrumb=breadcrumb,
        score_bps=min(score_bps, JEFFA_SCORE_BPS_CAP),
    )


def jeffa_snippet_to_text(snippet: SerpSnippet) -> str:
    parts = [snippet.title, snippet.display_url, snippet.description]
    if snippet.breadcrumb:
        parts.insert(1, snippet.breadcrumb)
    return " | ".join(parts)


# ------------------------------------------------------------------------------
# Page scoring (title, description, H1, keyword, length)
# ------------------------------------------------------------------------------

def jeffa_score_title(title: str, primary_keyword: str) -> int:
    if not title or not primary_keyword:
        return 0
    norm = jeffa_normalize_keyword(primary_keyword)
    title_lower = title.strip().lower()
    if norm in title_lower:
        pos = title_lower.index(norm)
        if pos < 30:
            return 9500
        if pos < 50:
            return 8000
        return 6000
    return 3000


def jeffa_score_description(description: str, primary_keyword: str) -> int:
    if not description or not primary_keyword:
        return 0
    norm = jeffa_normalize_keyword(primary_keyword)
    desc_lower = description.strip().lower()
    if norm in desc_lower:
        return 9000
    return 4000


def jeffa_score_h1(h1_list: list[str], primary_keyword: str) -> int:
    if not primary_keyword or not h1_list:
        return 5000
    norm = jeffa_normalize_keyword(primary_keyword)
    if len(h1_list) > JEFFA_H1_MAX_COUNT_RECOMMEND:
        return 4000
    for h in h1_list:
        if norm in h.strip().lower():
            return 9500
    return 5000


def jeffa_score_keyword_density(density_bps: int) -> int:
    if density_bps < int(JEFFA_KEYWORD_DENSITY_FLOOR * 10000):
        return 3000
    if density_bps > int(JEFFA_KEYWORD_DENSITY_CEIL * 10000):
        return 5000
    return 9000


def jeffa_score_length(word_count: int) -> int:
    if word_count < JEFFA_READABILITY_MIN_WORDS:
        return 4000
    if word_count >= JEFFA_READABILITY_IDEAL_WORDS:
        return 10000
    return 4000 + (6000 * (word_count - JEFFA_READABILITY_MIN_WORDS)) // (
        JEFFA_READABILITY_IDEAL_WORDS - JEFFA_READABILITY_MIN_WORDS
    )


def jeffa_grade_from_bps(bps: int) -> ContentGrade:
    if bps >= 90 * 100:
        return ContentGrade.A
    if bps >= 75 * 100:
        return ContentGrade.B
    if bps >= 60 * 100:
        return ContentGrade.C
    if bps >= 40 * 100:
        return ContentGrade.D
    return ContentGrade.F


def jeffa_page_score(
    title: str,
    description: str,
    body_text: str,
    h1_list: list[str],
    primary_keyword: str,
) -> PageScore:
    wc = len(jeffa_tokenize(body_text))
    kw_result = jeffa_analyze_keyword_in_text(body_text, primary_keyword) if primary_keyword else None
    density_bps = kw_result.density_bps if kw_result else 0

    t_bps = jeffa_score_title(title, primary_keyword)
    d_bps = jeffa_score_description(description, primary_keyword)
    h_bps = jeffa_score_h1(h1_list, primary_keyword)
    k_bps = jeffa_score_keyword_density(density_bps)
    l_bps = jeffa_score_length(wc)

    total_bps = (t_bps + d_bps + h_bps + k_bps + l_bps) // 5
    total_bps = min(total_bps, JEFFA_SCORE_BPS_CAP)
    grade = jeffa_grade_from_bps(total_bps)

    suggestions = []
    if t_bps < 6000:
        suggestions.append("Include primary keyword in title and keep under 60 chars.")
    if d_bps < 6000:
        suggestions.append("Include primary keyword in meta description (120–160 chars).")
    if h_bps < 7000:
        suggestions.append("Use one H1 containing the primary keyword.")
    if k_bps < 6000:
        suggestions.append("Adjust keyword density (target 0.5%–3%).")
    if l_bps < 6000:
        suggestions.append("Increase content length (aim for 800+ words for core topics).")

    return PageScore(
        total_bps=total_bps,
        title_score_bps=t_bps,
        desc_score_bps=d_bps,
        h1_score_bps=h_bps,
        keyword_score_bps=k_bps,
        length_score_bps=l_bps,
        grade=grade,
        suggestions=suggestions,
    )


# ------------------------------------------------------------------------------
# Sitemap generation (XML)
# ------------------------------------------------------------------------------

def jeffa_sitemap_ns() -> dict[str, str]:
    return {
        "": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "xhtml": "http://www.w3.org/1999/xhtml",
    }


def jeffa_sitemap_add_url(root: ET.Element, url: SitemapUrl, ns: dict[str, str]) -> None:
    url_el = ET.SubElement(root, "url")
    ET.SubElement(url_el, "loc").text = url.loc
    if url.lastmod:
        ET.SubElement(url_el, "lastmod").text = url.lastmod
    if url.changefreq:
        ET.SubElement(url_el, "changefreq").text = url.changefreq
    if url.priority is not None:
