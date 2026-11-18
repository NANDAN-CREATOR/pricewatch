"""Shopify stores: read the public product JSON (https://<store>/products/<handle>.json).

Why this matters for clothing: Shopify exposes `compare_at_price` (the struck-through "MRP")
next to `price`, so MRP inflation can be measured exactly. Many Indian D2C fashion brands and
multi-brand shops run on Shopify. Not every store leaves the endpoint open; those are reported
as errors. Same politeness rules as every source: robots.txt, delay, identified user agent.

In watchlist.json put product page URLs under "shopify": {"StoreName": "https://store.com/products/some-handle"}.
"""

from __future__ import annotations

import json
import re
import urllib.error
from urllib.parse import urlparse

from ..match import Product
from ..models import Offer
from ..normalize import parse_price
from ..polite import PoliteFetcher
from .base import Source

_HANDLE_RE = re.compile(r"/products/([^/?#]+)")


def json_url(product_url: str) -> str:
    """https://store.com/collections/x/products/abc?variant=1 -> https://store.com/products/abc.json"""
    parts = urlparse(product_url)
    m = _HANDLE_RE.search(parts.path)
    if not m:
        raise ValueError(f"not a Shopify product URL (no /products/<handle>): {product_url}")
    handle = m.group(1).removesuffix(".json")
    return f"{parts.scheme}://{parts.netloc}/products/{handle}.json"


def parse_product(data: dict, store: str, page_url: str = "", currency: str = "INR") -> Offer | None:
    """Turn one Shopify product JSON into an Offer using the cheapest available variant."""
    product = data.get("product", data)
    variants = product.get("variants") or []
    if not variants:
        return None
    available = [v for v in variants if v.get("available", True)]
    pool = available or variants
    priced = [(parse_price(v.get("price")), v) for v in pool]
    priced = [(p, v) for p, v in priced if p is not None]
    if not priced:
        return None
    price, variant = min(priced, key=lambda pv: pv[0])

    compare_at = parse_price(variant.get("compare_at_price"))
    mrp = compare_at if compare_at and compare_at > price else None

    title = str(product.get("title", "")).strip()
    vendor = str(product.get("vendor", "")).strip()
    if vendor and vendor.lower() not in title.lower():
        title = f"{vendor} {title}"  # brand often lives only in `vendor`
    image = (product.get("image") or {}).get("src") or next(
        (i.get("src") for i in product.get("images") or [] if i.get("src")), "")
    return Offer(store=store, title=title, price=price, mrp=mrp, url=page_url, currency=currency,
                 in_stock=bool(available), image_url=image or "")


class ShopifySource(Source):
    name = "shopify"

    def __init__(self, delay_seconds: float = 2.0, timeout: float = 15.0, currency: str = "INR"):
        self.fetcher = PoliteFetcher(delay_seconds, timeout)
        self.currency = currency

    def search(self, product: Product) -> list[Offer]:
        offers: list[Offer] = []
        errors: list[str] = []
        for store, url in product.shopify.items():
            try:
                body = self.fetcher.get(json_url(url), accept="application/json")
                offer = parse_product(json.loads(body), store, url, self.currency)
                if offer:
                    offers.append(offer)
                else:
                    errors.append(f"{store}: no priced variants")
            except (urllib.error.URLError, TimeoutError, OSError, PermissionError, ValueError) as exc:
                errors.append(f"{store}: {exc}")
        self.warnings = errors  # partial failures: surfaced on the page even when other stores worked
        if errors and not offers:
            raise RuntimeError("; ".join(errors))
        return offers
