"""Discovery via SerpApi's Google Shopping engine: one call returns listings from many stores
with price, struck-through price, store name and thumbnail. Needs SERPAPI_KEY."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from ..models import Offer
from ..normalize import parse_price
from . import SearchError

URL = "https://serpapi.com/search.json"


def _results(data: dict) -> list[dict]:
    rows = list(data.get("shopping_results") or [])
    for block in data.get("categorized_shopping_results") or []:
        rows.extend(block.get("shopping_results") or [])
    return rows


def parse_results(data: dict, currency: str = "INR") -> list[Offer]:
    """Turn a SerpApi google_shopping response into Offers. Pure function, easy to test."""
    offers: list[Offer] = []
    for r in _results(data):
        price = parse_price(r.get("extracted_price")) or parse_price(r.get("price"))
        title = str(r.get("title") or "").strip()
        if price is None or not title:
            continue
        condition = r.get("second_hand_condition")
        if condition:  # "refurbished", "used"... lets the matcher reject them like any other listing
            title = f"{title} ({condition})"
        old = parse_price(r.get("extracted_old_price")) or parse_price(r.get("old_price"))
        offers.append(
            Offer(
                store=str(r.get("source") or "Unknown store"),
                title=title,
                price=price,
                mrp=old if old and old > price else None,
                url=str(r.get("product_link") or r.get("link") or ""),
                currency=currency,
                image_url=str(r.get("thumbnail") or r.get("serpapi_thumbnail") or ""),
            )
        )
    return offers


class SerpApiShopping:
    name = "serpapi"

    def __init__(self, api_key: str | None = None, country: str = "in", language: str = "en", timeout: float = 25.0):
        self.api_key = (api_key or os.environ.get("SERPAPI_KEY", "")).strip().strip("'\"")  # tolerate pasted spaces/quotes
        self.country = country
        self.language = language
        self.timeout = timeout

    def search(self, query: str) -> list[Offer]:
        if not self.api_key:
            raise SearchError("Search is not configured: add a SERPAPI_KEY secret to this Space.")
        params = {"engine": "google_shopping", "q": query, "gl": self.country, "hl": self.language,
                  "api_key": self.api_key}
        req = urllib.request.Request(f"{URL}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "pricewatch/0.4"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SearchError("The search API key was rejected. Check the SERPAPI_KEY secret.") from None
            if exc.code == 429:
                raise SearchError("The search API quota is used up for now. Try again later.") from None
            raise SearchError(f"The search API returned an error (HTTP {exc.code}).") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise SearchError("Could not reach the search API. Try again in a moment.") from None
        if data.get("error") and not _results(data):
            raise SearchError(f"Search returned no results: {data['error']}")
        return parse_results(data, currency="INR" if self.country == "in" else "USD")
