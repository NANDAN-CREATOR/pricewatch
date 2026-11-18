import json
from datetime import datetime, timedelta, timezone

from pricewatch.inbox import GitHubInbox, ingest, make_command, make_record, product_id, send_command
from pricewatch.store import History
from pricewatch.tracker import build_trend, pick_to_check

T0 = datetime(2026, 11, 1, 9, tzinfo=timezone.utc)
OUT = {"query": "Levis 511 jeans", "summary": {"stores": [
    {"store": "A", "price": 2800.0, "url": "https://a"}, {"store": "B", "price": 3000.0}]}}


def test_product_id_ignores_order_and_case_and_keeps_gender():
    assert product_id("Levi's 511 Jeans") == product_id("jeans 511 LEVIS")
    assert product_id("kurti", "women") != product_id("kurti", "men")


def test_make_record_needs_a_summary():
    assert make_record({"query": "x", "summary": None}, None, "general") is None
    rec = make_record(OUT, "men", "clothing", now=T0)
    assert rec["id"].endswith("~men") and rec["stores"][0]["price"] == 2800.0


def test_inbox_disabled_without_token(monkeypatch):
    monkeypatch.delenv("GH_HISTORY_TOKEN", raising=False)
    box = GitHubInbox()
    assert not box.enabled
    box.save(make_record(OUT, None, "general"))  # no-op, no error


def test_ingest_builds_history_and_registry_and_counts_one_check_per_day(tmp_path):
    inbox = tmp_path / "inbox"; inbox.mkdir()
    for i, (hours, price) in enumerate([(0, 3000), (3, 2900), (48, 2800), (96, 2700)]):
        rec = make_record({**OUT, "summary": {"stores": [{"store": "A", "price": price, "url": ""}]}}, None, "general",
                          now=T0 + timedelta(hours=hours))
        (inbox / f"{i}.json").write_text(json.dumps(rec))
    (inbox / "bad.json").write_text("{nope")
    h, reg = History(tmp_path / "h.jsonl"), {}
    assert ingest(inbox, h, reg) == 4 and not list(inbox.glob("*.json"))
    p = next(iter(reg.values()))
    assert p["searches"] == 4
    t = build_trend(p, h, {"stores": []}, now=T0 + timedelta(days=5))
    assert t["checks"] == 3 and t["latest_lowest"] == 2700 and t["verdict"]["code"] == "low"  # two searches on day 1 = one check


def test_pick_to_check_only_tracked_skips_fresh_and_prefers_stale_then_popular():
    now = T0 + timedelta(days=3)
    reg = {"fresh": {"id": "fresh", "tracked": True, "last_seen": (now - timedelta(hours=2)).isoformat(), "searches": 99},
           "old_pop": {"id": "old_pop", "tracked": True, "last_seen": (T0).isoformat(), "searches": 9},
           "old_unpop": {"id": "old_unpop", "tracked": True, "last_seen": (T0).isoformat(), "searches": 1},
           "never": {"id": "never", "tracked": True, "searches": 0},
           "not_tracked": {"id": "not_tracked", "searches": 500}}
    ids = [p["id"] for p in pick_to_check(reg, 3, now)]
    assert ids == ["never", "old_pop", "old_unpop"]
    assert [p["id"] for p in pick_to_check(reg, 1, now)] == ["never"]  # budget caps how many are checked


def test_track_and_untrack_commands(tmp_path):
    inbox = tmp_path / "inbox"; inbox.mkdir()
    for i, cmd in enumerate(["track", "untrack", "track"]):
        (inbox / f"c{i}.json").write_text(json.dumps(make_command(cmd, "Levis 511 jeans", "men", "clothing", now=T0 + timedelta(minutes=i))))
    reg: dict = {}
    assert ingest(inbox, History(tmp_path / "h.jsonl"), reg) == 0 and not list(inbox.glob("*.json"))
    p = next(iter(reg.values()))
    assert p["tracked"] is True and p["gender"] == "men"
    (inbox / "u.json").write_text(json.dumps(make_command("untrack", "levis 511 jeans", "men", "clothing", now=T0)))
    ingest(inbox, History(tmp_path / "h.jsonl"), reg)
    assert reg[p["id"]]["tracked"] is False


def test_command_rejects_unknown_action_and_needs_token():
    import pytest
    with pytest.raises(ValueError):
        make_command("delete", "x", None, "general")
    with pytest.raises(RuntimeError):
        send_command(GitHubInbox(token=""), make_command("track", "x", None, "general"))
