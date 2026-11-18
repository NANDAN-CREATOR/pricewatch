"""Append-only price history (JSON Lines). Committed to git so history survives between runs."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Offer


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class History:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def record(self, product_id: str, offers: list[Offer], ts: datetime | None = None) -> None:
        stamp = (ts or now_utc()).isoformat(timespec="seconds")
        for o in offers:
            self.rows.append(
                {"ts": stamp, "product_id": product_id, "store": o.store, "price": o.price, "mrp": o.mrp}
            )

    def reference_price(self, product_id: str, days: int = 90, now: datetime | None = None) -> float | None:
        """Median selling price over the last `days` days, across all stores. None if no history."""
        cutoff = (now or now_utc()) - timedelta(days=days)
        prices = [
            r["price"]
            for r in self.rows
            if r["product_id"] == product_id and datetime.fromisoformat(r["ts"]) >= cutoff
        ]
        return statistics.median(prices) if prices else None

    def series(self, product_id: str, max_points: int = 90) -> list[dict]:
        """Daily lowest price for the product, for the sparkline."""
        by_day: dict[str, float] = {}
        for r in self.rows:
            if r["product_id"] != product_id:
                continue
            day = r["ts"][:10]
            by_day[day] = min(by_day.get(day, r["price"]), r["price"])
        return [{"date": d, "lowest": p} for d, p in sorted(by_day.items())][-max_points:]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
