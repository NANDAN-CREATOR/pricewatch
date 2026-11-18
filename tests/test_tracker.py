from datetime import datetime, timedelta, timezone

from pricewatch.models import Offer
from pricewatch.search import SearchAuthError
from pricewatch.store import History
from pricewatch.tracker import build_trend, run, verdict

T0 = datetime(2026, 11, 1, 9, tzinfo=timezone.utc)
PRODUCT = {"id": "p1", "name": "levis 511 slim fit jeans", "category": "clothing", "gender": "men"}


class Prov:
    name = "fake"

    def __init__(self, price_a, price_b):
        self.prices = (price_a, price_b)

    def search(self, q, country="in"):
        a, b = self.prices
        return [Offer("A", "Levi's 511 Slim Fit Jeans Men", a, url="https://a"),
                Offer("B", "Levis 511 Slim Fit Jeans Men", b, url="https://b")]


def test_verdict_needs_history_then_flags_low_and_high():
    pts = lambda *l: [{"lowest": x} for x in l]  # noqa: E731
    assert verdict(pts(100, 90))["code"] == "building"
    assert verdict(pts(120, 110, 100))["code"] == "low"
    assert verdict(pts(100, 110, 125))["code"] == "high"
    assert verdict(pts(100, 95, 102, 104))["code"] == "fair"


def test_run_records_history_and_shows_price_drop(tmp_path):
    h = History(tmp_path / "t.jsonl")
    d1 = run([PRODUCT], Prov(2800, 3000), h, now=T0)
    assert d1["products"][0]["checks"] == 1 and d1["products"][0]["latest_lowest"] == 2800
    d2 = run([PRODUCT], Prov(2500, 3000), h, now=T0 + timedelta(days=2))
    p = d2["products"][0]
    assert p["checks"] == 2 and p["latest_lowest"] == 2500 and p["period_high"] == 2800
    changes = {s["store"]: s["change_pct"] for s in p["stores"]}
    assert changes == {"A": -10.7, "B": 0.0}
    d3 = run([PRODUCT], Prov(2400, 3000), h, now=T0 + timedelta(days=4))
    assert d3["products"][0]["verdict"]["code"] == "low"  # third check, lowest so far


def test_failed_search_is_reported_not_raised(tmp_path):
    class Bad:
        name = "bad"
        def search(self, q, country="in"): raise SearchAuthError("key rejected")
    d = run([PRODUCT], Bad(), History(tmp_path / "t.jsonl"), now=T0)
    assert "key rejected" in d["products"][0]["error"] and d["products"][0]["points"] == []


def test_history_roundtrip_and_old_runs_expire(tmp_path):
    h = History(tmp_path / "t.jsonl")
    h.record("p1", [Offer("A", "x", 100)], ts=T0)
    h.record("p1", [Offer("A", "x", 90)], ts=T0 + timedelta(days=100))
    h.save()
    t = build_trend(PRODUCT, History(tmp_path / "t.jsonl"), None, now=T0 + timedelta(days=101))
    assert [p["lowest"] for p in t["points"]] == [90]  # the 101-day-old check is outside the window
