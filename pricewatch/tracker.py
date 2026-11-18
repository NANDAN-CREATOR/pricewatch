"""Scheduled price tracking: search each tracked product, record every store's price, and build the
trend data (history, lowest/median over time, price drops, a buy-or-wait verdict) that the site shows.

Run:  SEARCHAPI_KEY=... python -m pricewatch.tracker          (writes data/trends.jsonl, site/trends.json)
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .finder import Request, find
from .models import Offer
from .search import SearchError, SearchProvider
from .store import History

MIN_POINTS_FOR_VERDICT = 3
WINDOW_DAYS = 90


def verdict(points: list[dict]) -> dict:
    """Buy-or-wait signal from the lowest price over time. Honest about thin history."""
    if len(points) < MIN_POINTS_FOR_VERDICT:
        return {"code": "building", "text": f"Not enough history yet ({len(points)} check{'s' * (len(points) != 1)}). "
                                            f"A verdict appears after {MIN_POINTS_FOR_VERDICT} checks."}
    lows = [p["lowest"] for p in points]
    now, best, worst = lows[-1], min(lows), max(lows)
    over_best = (now - best) / best * 100
    if now <= best * 1.02:
        return {"code": "low", "text": "At or near the lowest price we have seen. A good time to buy."}
    if over_best >= 10:
        return {"code": "high", "text": f"{over_best:.0f}% above its lowest price so far. It may be worth waiting."}
    return {"code": "fair", "text": f"Within {over_best:.0f}% of its lowest price so far. A fair price."}


def build_trend(product: dict, history: History, latest: dict | None, now: datetime | None = None) -> dict:
    """Trend block for one product from recorded history plus this run's store details."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WINDOW_DAYS)
    runs: dict[str, list[dict]] = {}
    for r in history.rows:
        if r["product_id"] == product["id"] and datetime.fromisoformat(r["ts"]) >= cutoff:
            runs.setdefault(r["ts"], []).append(r)
    points = []
    for ts in sorted(runs):
        prices = [r["price"] for r in runs[ts]]
        points.append({"ts": ts, "date": ts[:10], "lowest": min(prices), "median": round(statistics.median(prices), 2),
                       "stores": len(prices)})
    # previous run's price per store, to show price drops since the last check
    prev_ts = sorted(runs)[-2] if len(runs) >= 2 else None
    prev = {r["store"]: r["price"] for r in runs[prev_ts]} if prev_ts else {}
    stores = []
    for s in (latest or {}).get("stores", []):
        old = prev.get(s["store"])
        change = round((s["price"] - old) / old * 100, 1) if old else None
        stores.append({**s, "change_pct": change})
    out = {"id": product["id"], "name": product["name"], "points": points, "verdict": verdict(points),
           "stores": stores, "error": (latest or {}).get("error")}
    if points:
        lows = [p["lowest"] for p in points]
        out.update(latest_date=points[-1]["date"], latest_lowest=points[-1]["lowest"], latest_median=points[-1]["median"],
                   period_low=min(lows), period_high=max(lows), checks=len(points))
    return out


def run(tracked: list[dict], provider: SearchProvider, history: History, now: datetime | None = None,
        limit: int = 5, previous: dict | None = None) -> dict:
    """Search each tracked product, record prices, and return the trends document."""
    now = now or datetime.now(timezone.utc)
    previous = previous or {}
    trends = []
    for p in tracked[:limit]:
        latest: dict = {}
        try:
            out = find(Request(query=p["name"], category=p.get("category", "general"), gender=p.get("gender")),
                       provider, None)
            summary = out["summary"]
            if summary:
                offers = [Offer(store=s["store"], title=s.get("title") or p["name"], price=s["price"], url=s.get("url", ""))
                          for s in summary["stores"]]
                history.record(p["id"], offers, ts=now)
                latest = {"stores": [{k: s.get(k) for k in ("store", "price", "url", "title", "image_url")}
                                     for s in summary["stores"]]}
            else:
                latest = {"error": "No listing could be confirmed as this product on this check."}
        except SearchError as exc:
            latest = {"error": str(exc)}
        if not latest.get("stores") and previous.get(p["id"], {}).get("stores"):  # keep last known store details
            latest["stores"] = [{k: s.get(k) for k in ("store", "price", "url", "title", "image_url")}
                                for s in previous[p["id"]]["stores"]]
        trends.append(build_trend(p, history, latest, now))
    return {"generated_at": now.isoformat(timespec="seconds"), "products": trends}


def main(argv: list[str] | None = None) -> int:
    from .search import FirstAccepted
    from .search.searchapi import SearchApiShopping
    from .search.serpapi import SerpApiShopping

    root = Path(".")
    tracked = json.loads((root / "tracked.json").read_text(encoding="utf-8"))["products"]
    provider = FirstAccepted([SearchApiShopping(), SerpApiShopping()])
    out_path = root / "site" / "trends.json"
    previous = {}
    if out_path.exists():
        previous = {p["id"]: p for p in json.loads(out_path.read_text(encoding="utf-8")).get("products", [])}
    history = History(root / "data" / "trends.jsonl")
    doc = run(tracked, provider, history, previous=previous)
    errors = [p for p in doc["products"] if p.get("error")]
    if len(errors) == len(doc["products"]):
        print("Every product failed:", "; ".join(p["error"] for p in errors), file=sys.stderr)
        return 1  # nothing new to commit; make the Action show red
    history.save()
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in doc["products"]:
        print(f"{p['name']}: {p.get('latest_lowest', '-')} ({p['verdict']['text']})" + (f"  ERROR {p['error']}" if p.get("error") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
