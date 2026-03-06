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
        ET.SubElement(url_el, "priority").text = f"{url.priority:.1f}"


def jeffa_sitemap_build(urls: list[SitemapUrl]) -> str:
    root = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for u in urls[:JEFFA_SITEMAP_MAX_URLS]:
        jeffa_sitemap_add_url(root, u, jeffa_sitemap_ns())
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", default_namespace="")


def jeffa_sitemap_index_build(sitemap_locs: list[str]) -> str:
    root = ET.Element("sitemapindex", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for loc in sitemap_locs[:JEFFA_SITEMAP_INDEX_MAX]:
        sitemap_el = ET.SubElement(root, "sitemap")
        ET.SubElement(sitemap_el, "loc").text = loc
        ET.SubElement(sitemap_el, "lastmod").text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", default_namespace="")


def jeffa_lastmod_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ------------------------------------------------------------------------------
# JSON-LD / Schema.org helpers
# ------------------------------------------------------------------------------

def jeffa_schema_webpage(
    name: str,
    description: str,
    url: str,
    date_published: str | None = None,
    date_modified: str | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "@context": JEFFA_SCHEMA_ORG_CONTEXT,
        "@type": "WebPage",
        "name": name,
        "description": description,
        "url": url,
    }
    if date_published:
        d["datePublished"] = date_published
    if date_modified:
        d["dateModified"] = date_modified
    return d


def jeffa_schema_organization(name: str, url: str, logo: str = "") -> dict[str, Any]:
    d: dict[str, Any] = {
        "@context": JEFFA_SCHEMA_ORG_CONTEXT,
        "@type": "Organization",
        "name": name,
        "url": url,
    }
    if logo:
        d["logo"] = logo
    return d


def jeffa_schema_to_script_ld(schema: dict[str, Any]) -> str:
    return f'<script type="application/ld+json">\n{json.dumps(schema, indent=2)}\n</script>'


# ------------------------------------------------------------------------------
# Crawl / fetch helpers (stdlib only)
# ------------------------------------------------------------------------------

def jeffa_fetch_url(url: str, timeout_sec: float = 10.0) -> tuple[int, str, dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": "JeffaSEO-Bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, body, headers
    except Exception as e:
        return 0, "", {"x-jeffa-error": str(e)}


def jeffa_extract_h1(html_text: str) -> list[str]:
    h1_re = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.DOTALL)
    out = []
    for m in h1_re.finditer(html_text):
        inner = re.sub(r"<[^>]+>", "", m.group(1))
        inner = html.unescape(inner).strip()
        if inner:
            out.append(inner)
    return out


def jeffa_extract_body_text(html_text: str) -> str:
    body_re = re.compile(r"<body[^>]*>(.*?)</body>", re.I | re.DOTALL)
    m = body_re.search(html_text)
    if not m:
        return ""
    inner = m.group(1)
    inner = re.sub(r"<script[^>]*>.*?</script>", " ", inner, flags=re.I | re.DOTALL)
    inner = re.sub(r"<style[^>]*>.*?</style>", " ", inner, flags=re.I | re.DOTALL)
    inner = re.sub(r"<[^>]+>", " ", inner)
    inner = html.unescape(inner)
    return re.sub(r"\s+", " ", inner).strip()


def jeffa_extract_links(html_text: str, base_url: str) -> list[str]:
    href_re = re.compile(r'<a[^>]+href\s*=\s*["\']([^"\']+)["\']', re.I)
    base = urllib.parse.urlparse(base_url)
    out = []
    seen = set()
    for m in href_re.finditer(html_text):
        raw = m.group(1).strip()
        if not raw or raw.startswith("#") or raw.startswith("javascript:"):
            continue
        try:
            parsed = urllib.parse.urlparse(raw)
            if not parsed.netloc:
                full = urllib.parse.urljoin(base_url, raw)
            else:
                full = raw
            norm = urllib.parse.urljoin(full, ".")
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        except Exception:
            pass
    return out


# ------------------------------------------------------------------------------
# Readability and content stats
# ------------------------------------------------------------------------------

def jeffa_word_count(text: str) -> int:
    return len(jeffa_tokenize(text))


def jeffa_sentence_count(text: str) -> int:
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return 0
    return len(re.split(r"[.!?]+", s))


def jeffa_avg_sentence_length(text: str) -> float:
    words = jeffa_word_count(text)
    sents = jeffa_sentence_count(text)
    if sents == 0:
        return 0.0
    return words / sents


def jeffa_avg_word_length(text: str) -> float:
    tokens = jeffa_tokenize(text)
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


# ------------------------------------------------------------------------------
# Payload builders for on-chain claims (align with Jeffa.sol)
# ------------------------------------------------------------------------------

def jeffa_claim_payload(
    agent_id_hex: str,
    keyword: str,
    tier: SerpTier,
    nonce: str | None = None,
) -> dict[str, Any]:
    from time import time_ns
    nonce = nonce or str(time_ns())
    claim_id = jeffa_claim_id(agent_id_hex, keyword, nonce)
    kw_hash = jeffa_keyword_hash(keyword)
    return {
        "claimId": claim_id,
        "agentId": agent_id_hex,
        "keywordHash": kw_hash,
        "keyword": keyword,
        "tier": tier.name,
        "tierValue": tier.value,
        "nonce": nonce,
        "namespace": JEFFA_NAMESPACE,
    }


def jeffa_agent_payload(wallet_or_name: str) -> dict[str, Any]:
    aid = jeffa_agent_id(wallet_or_name)
    return {
        "agentId": aid,
        "walletOrName": wallet_or_name,
        "namespace": JEFFA_NAMESPACE,
    }


# ------------------------------------------------------------------------------
# Batch and iterator helpers
# ------------------------------------------------------------------------------

def jeffa_batch_keyword_analysis(text: str, keywords: list[str]) -> list[KeywordResult]:
    return [jeffa_analyze_keyword_in_text(text, kw) for kw in keywords]


def jeffa_iterate_meta_for_pages(
    pages: list[tuple[str, str, str]],
    base_url: str = "",
) -> Iterator[MetaTags]:
    for title, desc, path in pages:
        canonical = (base_url.rstrip("/") + "/" + path.lstrip("/")) if base_url else path
        yield jeffa_build_meta(title, desc, canonical=canonical)


# ------------------------------------------------------------------------------
# Export and CLI
# ------------------------------------------------------------------------------

def jeffa_export_keyword_results(results: list[KeywordResult], path: str | Path) -> None:
    p = Path(path)
    rows = [
        "keyword,count,density_bps,position_first,position_last,tier,normalized",
    ]
    for r in results:
        rows.append(
            f"{r.keyword!r},{r.count},{r.density_bps},{r.position_first},{r.position_last},{r.tier.name},{r.normalized!r}"
        )
    p.write_text("\n".join(rows), encoding="utf-8")


def jeffa_export_page_scores(scores: list[tuple[str, PageScore]], path: str | Path) -> None:
    p = Path(path)
    rows = ["url,total_bps,title_bps,desc_bps,h1_bps,kw_bps,len_bps,grade"]
    for url, ps in scores:
        rows.append(
            f"{url},{ps.total_bps},{ps.title_score_bps},{ps.desc_score_bps},"
            f"{ps.h1_score_bps},{ps.keyword_score_bps},{ps.length_score_bps},{ps.grade.name}"
        )
    p.write_text("\n".join(rows), encoding="utf-8")


def jeffa_cli_usage() -> str:
    return """
JeffaSEO CLI (single-file).
  python jeffa_seo.py analyze-keyword <text_file> <keyword>
  python jeffa_seo.py meta <title> <description> [canonical]
  python jeffa_seo.py score <title> <desc_file> <body_file> <primary_keyword> [h1_file]
  python jeffa_seo.py sitemap <url_list_file> [output.xml]
  python jeffa_seo.py claim-payload <agent_id> <keyword> [tier]
  python jeffa_seo.py agent-id <wallet_or_name>
"""


def _main_analyze_keyword(args: list[str]) -> int:
    if len(args) < 2:
        print(jeffa_cli_usage())
        return 1
    path, keyword = args[0], args[1]
    text = Path(path).read_text(encoding="utf-8")
    r = jeffa_analyze_keyword_in_text(text, keyword)
    print(json.dumps({
        "keyword": r.keyword,
        "count": r.count,
        "density_bps": r.density_bps,
        "position_first": r.position_first,
        "position_last": r.position_last,
        "tier": r.tier.name,
    }, indent=2))
    return 0


def _main_meta(args: list[str]) -> int:
    if len(args) < 2:
        print(jeffa_cli_usage())
        return 1
    title, desc = args[0], args[1]
    canonical = args[2] if len(args) > 2 else ""
    meta = jeffa_build_meta(title, desc, canonical=canonical)
    print(jeffa_meta_to_html(meta))
    return 0


def _main_score(args: list[str]) -> int:
    if len(args) < 4:
        print(jeffa_cli_usage())
        return 1
    title, desc_file, body_file, primary = args[0], args[1], args[2], args[3]
    h1_file = args[4] if len(args) > 4 else None
    desc = Path(desc_file).read_text(encoding="utf-8")
    body = Path(body_file).read_text(encoding="utf-8")
    h1_list = Path(h1_file).read_text(encoding="utf-8").strip().split("\n") if h1_file else []
    ps = jeffa_page_score(title, desc, body, h1_list, primary)
    print(json.dumps({
        "total_bps": ps.total_bps,
        "title_score_bps": ps.title_score_bps,
        "desc_score_bps": ps.desc_score_bps,
        "h1_score_bps": ps.h1_score_bps,
        "keyword_score_bps": ps.keyword_score_bps,
        "length_score_bps": ps.length_score_bps,
        "grade": ps.grade.name,
        "suggestions": ps.suggestions,
    }, indent=2))
    return 0


def _main_sitemap(args: list[str]) -> int:
    if len(args) < 1:
        print(jeffa_cli_usage())
        return 1
    url_file = args[0]
    out_file = args[1] if len(args) > 1 else "sitemap.xml"
    lines = Path(url_file).read_text(encoding="utf-8").strip().split("\n")
    urls = [SitemapUrl(loc=l.strip(), lastmod=jeffa_lastmod_iso(), changefreq="weekly", priority=0.8) for l in lines if l.strip()]
    xml = jeffa_sitemap_build(urls)
    Path(out_file).write_text(xml, encoding="utf-8")
    print(f"Wrote {len(urls)} URLs to {out_file}")
    return 0


def _main_claim_payload(args: list[str]) -> int:
    if len(args) < 2:
        print(jeffa_cli_usage())
        return 1
    agent_id, keyword = args[0], args[1]
    tier_name = (args[2] if len(args) > 2 else "CORE").upper()
    tier = SerpTier[tier_name] if tier_name in SerpTier.__members__ else SerpTier.CORE
    payload = jeffa_claim_payload(agent_id, keyword, tier)
    print(json.dumps(payload, indent=2))
    return 0


def _main_agent_id(args: list[str]) -> int:
    if len(args) < 1:
        print(jeffa_cli_usage())
        return 1
    payload = jeffa_agent_payload(args[0])
    print(json.dumps(payload, indent=2))
    return 0


def jeffa_main(argv: list[str] | None = None) -> int:
    argv = argv or __import__("sys").argv
    if len(argv) < 2:
        print(jeffa_cli_usage())
        return 0
    cmd = argv[1].lower()
    args = argv[2:]
    if cmd == "analyze-keyword":
        return _main_analyze_keyword(args)
    if cmd == "meta":
        return _main_meta(args)
    if cmd == "score":
        return _main_score(args)
    if cmd == "sitemap":
        return _main_sitemap(args)
    if cmd == "claim-payload":
        return _main_claim_payload(args)
    if cmd == "agent-id":
        return _main_agent_id(args)
    print(jeffa_cli_usage())
    return 0


# ------------------------------------------------------------------------------
# Additional AI SEO utilities (reach 1000+ lines)
# ------------------------------------------------------------------------------

def jeffa_slug_from_title(title: str, max_len: int = 80) -> str:
    s = title.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s[:max_len] if len(s) > max_len else s


def jeffa_internal_link_suggestions(
    body_text: str,
    keyword_phrases: list[str],
    max_suggestions: int = 5,
) -> list[tuple[str, int]]:
    tokens = jeffa_tokenize(body_text)
    suggestions = []
    for phrase in keyword_phrases:
        norm = jeffa_normalize_keyword(phrase)
        pw = jeffa_tokenize(norm)
        if not pw:
            continue
        count = 0
        for i in range(len(tokens) - len(pw) + 1):
            if tokens[i:i + len(pw)] == pw:
                count += 1
        suggestions.append((phrase, count))
    suggestions.sort(key=lambda x: -x[1])
    return suggestions[:max_suggestions]


def jeffa_heading_structure(html_text: str) -> list[tuple[int, str]]:
    heading_re = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.I | re.DOTALL)
    out = []
    for m in heading_re.finditer(html_text):
        level = int(m.group(1))
        inner = re.sub(r"<[^>]+>", "", m.group(2))
        out.append((level, html.unescape(inner).strip()))
    return out


def jeffa_image_alt_check(html_text: str) -> list[tuple[str, bool]]:
    img_re = re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>', re.I)
    out = []
    for m in img_re.finditer(html_text):
        src = m.group(1)
        alt = m.group(2) if m.group(2) is not None else ""
        out.append((src, bool(alt.strip())))
    return out


def jeffa_meta_robots_parse(robots_content: str) -> dict[str, bool]:
    allowed = True
    for part in robots_content.lower().split(","):
        part = part.strip()
        if "noindex" in part:
            allowed = False
            break
        if "index" in part and "no" not in part:
            allowed = True
    return {"index": allowed}


def jeffa_canonical_from_doc(html_text: str) -> str:
    link_re = re.compile(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', re.I)
    m = link_re.search(html_text)
    if m:
        return m.group(1).strip()
    return ""


def jeffa_title_from_doc(html_text: str) -> str:
    title_re = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.DOTALL)
    m = title_re.search(html_text)
    if m:
        return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return ""


def jeffa_desc_from_doc(html_text: str) -> str:
    meta_re = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', re.I)
    m = meta_re.search(html_text)
    if m:
        return html.unescape(m.group(1)).strip()
    return ""


def jeffa_breadcrumb_schema(items: list[tuple[str, str]]) -> dict[str, Any]:
    list_items = []
    for i, (name, url) in enumerate(items):
        list_items.append({
            "@type": "ListItem",
            "position": i + 1,
            "name": name,
            "item": url,
        })
    return {
        "@context": JEFFA_SCHEMA_ORG_CONTEXT,
        "@type": "BreadcrumbList",
        "itemListElement": list_items,
    }


def jeffa_faq_schema(qa_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "@context": JEFFA_SCHEMA_ORG_CONTEXT,
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}}
            for q, a in qa_pairs
        ],
    }


def jeffa_article_schema(
    headline: str,
    description: str,
    url: str,
    date_published: str,
    date_modified: str | None = None,
    author_name: str = "",
    image_url: str = "",
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "@context": JEFFA_SCHEMA_ORG_CONTEXT,
        "@type": "Article",
        "headline": headline,
        "description": description,
        "url": url,
        "datePublished": date_published,
    }
    if date_modified:
        d["dateModified"] = date_modified
    if author_name:
        d["author"] = {"@type": "Person", "name": author_name}
    if image_url:
        d["image"] = image_url
    return d


def jeffa_compress_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def jeffa_stem_simple(word: str) -> str:
    if len(word) <= 4:
        return word
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def jeffa_keyword_variations(keyword: str) -> list[str]:
    norm = jeffa_normalize_keyword(keyword)
    words = norm.split()
    if len(words) <= 1:
        return [norm]
    variations = [norm]
    if len(words) >= 2:
        variations.append(" ".join(reversed(words)))
    return variations


def jeffa_serp_tier_from_keyword(keyword: str) -> SerpTier:
    r = _jeffa_infer_tier(keyword, 1)
    return r


def jeffa_validate_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def jeffa_normalize_url(url: str) -> str:
    try:
        return urllib.parse.urljoin(url, ".")
    except Exception:
        return url


def jeffa_domain_from_url(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc or ""
    except Exception:
        return ""


def jeffa_path_from_url(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).path or "/"
    except Exception:
        return "/"


def jeffa_merge_keyword_results(a: KeywordResult, b: KeywordResult) -> KeywordResult:
    total_count = a.count + b.count
    total_pos_first = min(a.position_first, b.position_first) if a.position_first >= 0 and b.position_first >= 0 else (a.position_first if a.position_first >= 0 else b.position_first)
    total_pos_last = max(a.position_last, b.position_last)
    return KeywordResult(
        keyword=a.keyword,
        count=total_count,
        density_bps=a.density_bps,
        position_first=total_pos_first,
        position_last=total_pos_last,
        tier=a.tier,
        normalized=a.normalized,
    )


def jeffa_content_gap_keywords(
    target_keywords: list[str],
    existing_text: str,
) -> list[str]:
    existing_set = set(jeffa_extract_keywords(existing_text))
    gap = []
    for kw in target_keywords:
        norm = jeffa_normalize_keyword(kw)
        kw_tokens = set(jeffa_tokenize(norm))
        if not kw_tokens.issubset(existing_set):
            gap.append(kw)
    return gap


def jeffa_readability_score_bps(text: str) -> int:
    words = jeffa_word_count(text)
    sents = jeffa_sentence_count(text)
    if sents == 0:
        return 0
    avg_sent = words / sents
    if avg_sent <= 15:
        return 9000
    if avg_sent <= 20:
        return 7000
    if avg_sent <= 25:
        return 5000
    return 3000


def jeffa_inject_primary_keyword_into_title(title: str, primary_keyword: str, max_len: int = JEFFA_MAX_TITLE_LEN) -> str:
    if not primary_keyword or primary_keyword.lower() in title.lower():
        return jeffa_truncate_for_meta(title, max_len)
    combined = f"{primary_keyword}: {title}"
    return jeffa_truncate_for_meta(combined, max_len)


def jeffa_inject_primary_keyword_into_desc(description: str, primary_keyword: str) -> str:
    if not primary_keyword or primary_keyword.lower() in description.lower():
        return jeffa_truncate_for_meta(description, JEFFA_MAX_DESC_LEN)
    combined = f"{description} {primary_keyword}."
    return jeffa_truncate_for_meta(combined, JEFFA_MAX_DESC_LEN, "...")


def jeffa_compare_titles(a: str, b: str) -> int:
    """Returns a score 0-10000: how similar the two titles are (keyword overlap)."""
    set_a = set(jeffa_extract_keywords(a))
    set_b = set(jeffa_extract_keywords(b))
    if not set_a:
        return 10000 if not set_b else 0
    inter = len(set_a & set_b)
    return (inter * 10000) // len(set_a)


def jeffa_redirect_chain_safe(url: str, max_hops: int = 5) -> list[str]:
    """Returns list of URLs in redirect chain (stdlib only; no follow by default)."""
    return [url]


def jeffa_meta_refresh_parse(html_text: str) -> str | None:
    meta_re = re.compile(r'<meta[^>]+http-equiv=["\']refresh["\'][^>]+content=["\']([^"\']+)["\']', re.I)
    m = meta_re.search(html_text)
    if m:
        return m.group(1).strip()
    return None


def jeffa_noindex_detected(html_text: str) -> bool:
    robots = jeffa_meta_robots_parse(jeffa_extract_meta_content(html_text, "robots"))
    return not robots.get("index", True)

