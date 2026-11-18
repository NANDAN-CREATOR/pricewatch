"""Read prices from product pages via schema.org JSON-LD (what most stores publish for SEO).

You list product page URLs per store in watchlist.json ("urls"). This source:
  - checks robots.txt and skips pages that disallow us,
  - waits between requests,
  - extracts Product -> offers -> price from <script type="application/ld+json">.

It does not bypass logins, captchas or bot protection. Stores that block datacenter IPs
(e.g. GitHub Actions runners) will simply show up as a per-source error.
"""

from __future__ import annotations

import json
import re
import urllib.error
from html.parser import HTMLParser
from urllib.parse import urljoin

from ..match import Product
from ..models import Offer
from ..normalize import parse_price
from ..polite import PoliteFetcher
from .base import Source

_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE
)


def _walk(node):
    """Yield every dict inside a JSON-LD structure (handles lists and @graph)."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _is_type(node: dict, name: str) -> bool:
    t = node.get("@type")
    return t == name or (isinstance(t, list) and name in t)


def _image_of(node: dict) -> str:
    img = node.get("image")
    if isinstance(img, list):
        img = img[0] if img else ""
    if isinstance(img, dict):
        img = img.get("url", "")
    return str(img or "")


class _MetaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            key = a.get("property") or a.get("name")
            if key and a.get("content") is not None:
                self.meta.setdefault(key.lower(), a["content"])


def parse_meta(html: str) -> dict[str, str]:
    """<meta property/name=... content=...> as a dict (keys lower-cased). Used for OpenGraph fallbacks."""
    parser = _MetaParser()
    parser.feed(html)
    return parser.meta


def parse_jsonld(html: str, store: str, url: str = "") -> list[Offer]:
    """Extract offers from a page's HTML. Pure function, easy to test."""
    offers: list[Offer] = []
    for block in _LD_RE.findall(html):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for node in _walk(data):
            if not _is_type(node, "Product"):
                continue
            title = str(node.get("name", "")).strip()
            raw_offers = node.get("offers")
            if isinstance(raw_offers, dict):
                raw_offers = [raw_offers]
            for ro in raw_offers or []:
                if not isinstance(ro, dict):
                    continue
                price = parse_price(ro.get("price") or ro.get("lowPrice"))
                if not title or price is None:
                    continue
                avail = str(ro.get("availability", "InStock"))
                offers.append(
                    Offer(
                        store=store,
                        title=title,
                        price=price,
                        url=str(ro.get("url") or url),
                        currency=str(ro.get("priceCurrency") or "INR"),
                        in_stock="OutOfStock" not in avail and "SoldOut" not in avail,
                        image_url=urljoin(url, _image_of(node)) if _image_of(node) else "",
                    )
                )
    return offers


class JsonLdSource(Source):
    name = "jsonld"

    def __init__(self, delay_seconds: float = 2.0, timeout: float = 15.0):
        self.fetcher = PoliteFetcher(delay_seconds, timeout)

    def search(self, product: Product) -> list[Offer]:
        offers: list[Offer] = []
        errors: list[str] = []
        for store, url in product.urls.items():
            try:
                offers.extend(parse_jsonld(self.fetcher.get(url), store, url))
            except (urllib.error.URLError, TimeoutError, OSError, PermissionError) as exc:
                errors.append(f"{store}: {exc}")
        self.warnings = errors  # partial failures: surfaced on the page even when other stores worked
        if errors and not offers:
            raise RuntimeError("; ".join(errors))
        return offers
