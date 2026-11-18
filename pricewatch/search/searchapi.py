"""Discovery via SearchApi.io's Google Shopping engine (a different company from SerpApi).

Free tier available. Its shopping results carry the store in `seller`, a `durability` field for
used items, and no struck-through price, so MRP inflation is not available with this provider.
Key: SEARCHAPI_KEY (falls back to SERPAPI_KEY, since people often store it under that name).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from ..models import Offer
from ..normalize import parse_price
from . import SearchAuthError, SearchError

URL = "https://www.searchapi.io/api/v1/search"


def parse_results(data: dict, currency: str = "INR") -> list[Offer]:
    offers: list[Offer] = []
    for r in data.get("shopping_results") or []:
        price = parse_price(r.get("extracted_price")) or parse_price(r.get("price"))
        title = str(r.get("title") or "").strip()
        if price is None or not title:
            continue
        if r.get("durability"):  # e.g. "Pre-owned": lets the matcher reject it like any used listing
            title = f"{title} ({r['durability']})"
        old = parse_price(r.get("extracted_original_price")) or parse_price(r.get("original_price"))
        offers.append(
            Offer(
                store=str(r.get("seller") or r.get("source") or "Unknown store"),
                title=title,
                price=price,
                mrp=old if old and old > price else None,
                url=str(r.get("product_link") or r.get("offers_link") or ""),
                currency=currency,
                image_url=str(r.get("thumbnail") or ""),
            )
        )
    return offers


class SearchApiShopping:
    name = "searchapi"

    def __init__(self, api_key: str | None = None, country: str = "in", language: str = "en", timeout: float = 40.0):
        key = api_key or os.environ.get("SEARCHAPI_KEY") or os.environ.get("SERPAPI_KEY", "")
        self.api_key = key.strip().strip("'\"")
        self.country, self.language, self.timeout = country, language, timeout

    def search(self, query: str) -> list[Offer]:
        if not self.api_key:
            raise SearchError("Search is not configured: add a SEARCHAPI_KEY secret to this Space.")
        params = {"engine": "google_shopping", "q": query, "gl": self.country, "hl": self.language}
        req = urllib.request.Request(
            f"{URL}?{urllib.parse.urlencode(params)}",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json",
                     "User-Agent": "pricewatch/0.4"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SearchAuthError("The search API key was rejected by SearchApi.") from None
            if exc.code == 429:
                raise SearchError("The search API quota is used up for now. Try again later.") from None
            raise SearchError(f"The search API returned an error (HTTP {exc.code}).") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise SearchError("Could not reach the search API. Try again in a moment.") from None
        if data.get("error") and not data.get("shopping_results"):
            raise SearchError(f"Search returned no results: {data['error']}")
        return parse_results(data, currency="INR" if self.country == "in" else "USD")
