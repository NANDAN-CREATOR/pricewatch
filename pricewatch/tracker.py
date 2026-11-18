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
    by_ts: dict[str, list[dict]] = {}
    for r in history.rows:
        if r["product_id"] == product["id"] and datetime.fromisoformat(r["ts"]) >= cutoff:
            by_ts.setdefault(r["ts"], []).append(r)
    last_of_day: dict[str, str] = {}
    for ts in sorted(by_ts):
        last_of_day[ts[:10]] = ts          # several readings in one day count as one check
    runs = {ts: by_ts[ts] for ts in last_of_day.values()}
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
        limit: int = 5, previous: dict | None = None, show: list[dict] | None = None) -> dict:
    """Search each product in `tracked[:limit]`, record prices, and return the trends document for `show`
    (default: the products just checked)."""
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
    if show is not None:
        done = {t["id"]: t for t in trends}
        trends = []
        for p in show:
            if p["id"] in done:
                trends.append(done[p["id"]])
            else:
                old = previous.get(p["id"], {}).get("stores") or p.get("stores") or []
                trends.append(build_trend(p, history, {"stores": old}, now))
    return {"generated_at": now.isoformat(timespec="seconds"), "products": trends}


def pick_to_check(registry: dict, budget: int, now: datetime, fresh_hours: int = 6) -> list[dict]:
    """Products to re-check this run: skip ones already read recently (a visitor's search counts), then the
    least recently read first, most-searched first among ties."""
    cutoff = (now - timedelta(hours=fresh_hours)).isoformat(timespec="seconds")
    due = [p for p in registry.values() if p.get("last_seen", "") < cutoff]
    due.sort(key=lambda p: (p.get("last_seen", ""), -p.get("searches", 0)))
    return due[:budget]


def main(argv: list[str] | None = None) -> int:
    import os

    from .inbox import ingest, product_id
    from .search import FirstAccepted
    from .search.searchapi import SearchApiShopping
    from .search.serpapi import SerpApiShopping

    root = Path(".")
    now = datetime.now(timezone.utc)
    budget = int(os.environ.get("TRACK_BUDGET", "8"))
    history = History(root / "data" / "trends.jsonl")
    reg_path = root / "data" / "products.json"
    registry: dict = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}

    for p in json.loads((root / "tracked.json").read_text(encoding="utf-8"))["products"]:  # always-tracked seeds
        pid = product_id(p["name"], p.get("gender"))
        registry.setdefault(pid, {"id": pid, "name": p["name"], "category": p.get("category", "general"),
                                  "gender": p.get("gender"), "searches": 0})
    used = ingest(root / "data" / "inbox", history, registry)
    print(f"Folded {used} visitor searches into the history.")

    out_path = root / "site" / "trends.json"
    previous = {}
    if out_path.exists():
        previous = {p["id"]: p for p in json.loads(out_path.read_text(encoding="utf-8")).get("products", [])}
    check = pick_to_check(registry, budget, now)
    doc = {"products": []}
    errors: list[dict] = []
    if check:
        provider = FirstAccepted([SearchApiShopping(), SerpApiShopping()])
        doc = run(check, provider, history, now=now, limit=len(check), previous=previous)
        errors = [p for p in doc["products"] if p.get("error")]
        for p in doc["products"]:
            registry[p["id"]]["error"] = p.get("error")
            if not p.get("error"):
                registry[p["id"]]["last_seen"] = now.isoformat(timespec="seconds")
                registry[p["id"]]["stores"] = [{k: s.get(k) for k in ("store", "price", "url")} for s in p["stores"]]
    if check and len(errors) == len(check) and not used:
        print("Every product failed:", "; ".join(p["error"] for p in errors), file=sys.stderr)
        return 1  # nothing new to commit; make the Action show red
    history.save()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")

    shown = sorted(registry.values(), key=lambda p: -p.get("searches", 0))[:300]
    full = run([], None, history, now=now, previous=previous, show=shown)
    for t in full["products"]:
        t["error"] = registry[t["id"]].get("error") or t.get("error")
        t["searches"] = registry[t["id"]].get("searches", 0)
        t["points"] = t["points"][-60:]
    out_path.write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
    for p in doc["products"]:
        print(f"{p['name']}: {p.get('latest_lowest', '-')} ({p['verdict']['text']})" + (f"  ERROR {p['error']}" if p.get("error") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
