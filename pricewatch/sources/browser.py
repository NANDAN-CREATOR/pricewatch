"""Headless-browser scraper (Playwright) for pages that need JavaScript to show the price.

What it does: opens each product page in a real Chromium, waits for it to render, then reads
title / price / MRP / image. Extraction order, per page:
  1. CSS selectors you configured for that store (most precise),
  2. schema.org JSON-LD found in the rendered page,
  3. OpenGraph meta tags (product:price:amount, og:title, og:image).

What it deliberately does not do: hide that it is a bot, rotate proxies, or solve captchas.
It identifies itself, obeys robots.txt and waits between pages. A store that blocks it will show
up as a per-store error. Check each site's terms before you point this at it.

Needs: pip install playwright && playwright install chromium   (or set PRICEWATCH_CHROME to a Chrome binary)
"""

from __future__ import annotations

import os
import time
from urllib.parse import urljoin

from ..match import Product
from ..models import Offer
from ..normalize import parse_price
from ..polite import USER_AGENT, PoliteFetcher
from .base import Source
from .jsonld import parse_jsonld, parse_meta


def _text(page, selector: str | None) -> str | None:
    if not selector:
        return None
    try:
        return page.locator(selector).first.inner_text(timeout=3000).strip()
    except Exception:
        return None


def _attr(page, selector: str | None, attr: str) -> str | None:
    if not selector:
        return None
    try:
        return page.locator(selector).first.get_attribute(attr, timeout=3000)
    except Exception:
        return None


def extract(page, cfg: dict, store: str, url: str, currency: str = "INR") -> Offer | None:
    """Read one offer from an already-loaded Playwright page."""
    title = _text(page, cfg.get("title"))
    price = parse_price(_text(page, cfg.get("price")))
    mrp = parse_price(_text(page, cfg.get("mrp")))
    image = _attr(page, cfg.get("image"), cfg.get("image_attr", "src")) or ""
    in_stock = True

    html = page.content()
    if not (title and price):
        found = parse_jsonld(html, store, url)  # fallback 2: JSON-LD in the rendered DOM
        if found:
            first = found[0]
            title, price = title or first.title, price or first.price
            image, in_stock = image or first.image_url, first.in_stock
            currency = first.currency or currency
    if not (title and price):
        meta = parse_meta(html)  # fallback 3: OpenGraph
        title = title or meta.get("og:title")
        price = price or parse_price(meta.get("product:price:amount") or meta.get("og:price:amount"))
        image = image or meta.get("og:image", "")
        currency = meta.get("product:price:currency") or currency
    if not (title and price):
        return None

    sold_out = cfg.get("sold_out_text")
    if sold_out and sold_out.lower() in html.lower():
        in_stock = False
    if mrp is not None and mrp <= price:
        mrp = None
    return Offer(store=store, title=title, price=price, mrp=mrp, url=url, currency=currency,
                 in_stock=in_stock, image_url=urljoin(url, image) if image else "")


class BrowserSource(Source):
    name = "browser"

    def __init__(self, stores: dict | None = None, delay_seconds: float = 3.0, timeout_ms: int = 30000,
                 currency: str = "INR", executable_path: str | None = None, settle_ms: int = 1500):
        self.stores = stores or {}  # store name -> {"title","price","mrp","image","image_attr","sold_out_text"}
        self.delay = delay_seconds
        self.timeout_ms = timeout_ms
        self.currency = currency
        self.executable_path = executable_path or os.environ.get("PRICEWATCH_CHROME") or None
        self.settle_ms = settle_ms  # extra wait for client-side rendering
        self.fetcher = PoliteFetcher(delay_seconds)  # only used for the robots.txt check

    def search(self, product: Product) -> list[Offer]:
        if not product.pages:
            return []
        from playwright.sync_api import sync_playwright  # imported lazily: optional dependency

        offers: list[Offer] = []
        errors: list[str] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, executable_path=self.executable_path)
            try:
                page = browser.new_context(user_agent=USER_AGENT, locale="en-IN").new_page()
                for i, (store, url) in enumerate(product.pages.items()):
                    if i:
                        time.sleep(self.delay)
                    try:
                        if not self.fetcher.allowed(url):
                            errors.append(f"{store}: blocked by robots.txt (or unreachable)")
                            continue
                        resp = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                        if resp is not None and resp.status >= 400:
                            errors.append(f"{store}: HTTP {resp.status}")
                            continue
                        cfg = self.stores.get(store, {})
                        if cfg.get("price"):
                            try:
                                page.wait_for_selector(cfg["price"], timeout=self.timeout_ms // 2)
                            except Exception:
                                pass
                        else:
                            page.wait_for_timeout(self.settle_ms)
                        offer = extract(page, cfg, store, url, self.currency)
                        if offer:
                            offers.append(offer)
                        else:
                            errors.append(f"{store}: could not find a price (add selectors for this store)")
                    except Exception as exc:
                        errors.append(f"{store}: {exc}")
            finally:
                browser.close()
        self.warnings = errors  # partial failures: surfaced on the page even when other stores worked
        if errors and not offers:
            raise RuntimeError("; ".join(errors))
        return offers
