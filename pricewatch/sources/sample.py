"""Offline demo source that reads fixtures/sample_offers.json. Used for tests and the live demo."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..match import Product
from ..models import Offer
from .base import Source


class SampleSource(Source):
    name = "sample"

    def __init__(self, path: str = "fixtures/sample_offers.json", jitter: float = 0.0, seed: int | None = None):
        self.path = Path(path)
        self.jitter = jitter  # e.g. 0.03 moves each price by up to ±3%, to simulate daily drift
        self.rng = random.Random(seed)

    def search(self, product: Product) -> list[Offer]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        offers = []
        for row in data.get(product.id, []):
            price = row["price"]
            mrp = row.get("mrp")
            if self.jitter:
                factor = 1 + self.rng.uniform(-self.jitter, self.jitter)
                price = round(price * factor)
            offers.append(
                Offer(
                    store=row["store"],
                    title=row["title"],
                    price=float(price),
                    mrp=float(mrp) if mrp else None,
                    url=row.get("url", ""),
                    in_stock=row.get("in_stock", True),
                )
            )
        return offers
