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
