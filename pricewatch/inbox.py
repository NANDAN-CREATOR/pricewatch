"""Turn every search into price history.

The Space writes one small JSON file per search into `data/inbox/` in the GitHub repo (a new file each time,
so there are no write conflicts). The scheduled tracker later folds those files into `data/trends.jsonl`,
keeps a registry of products people searched (`data/products.json`) and re-checks the most-searched ones.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .models import Offer
from .normalize import tokenize
from .store import History

PUSH_COOLDOWN_S = 6 * 3600     # the same product is saved at most every 6 hours
MAX_PUSHES_PER_DAY = 300


def product_id(query: str, gender: str | None = None) -> str:
    """Stable id from a product name: word order and case do not matter."""
    toks = sorted(set(tokenize(query)))
    base = "-".join(toks)[:80].strip("-") or "unknown"
    return f"{base}~{gender}" if gender in ("men", "women") else base


def make_record(out: dict, gender: str | None, category: str, now: datetime | None = None) -> dict | None:
    s = out.get("summary")
    if not s or not out.get("query"):
        return None
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return {"ts": ts, "name": out["query"], "gender": gender, "category": category,
            "id": product_id(out["query"], gender),
            "stores": [{"store": r["store"], "price": r["price"], "url": r.get("url", "")} for r in s["stores"]]}


class GitHubInbox:
    """Saves search records to the repo. Does nothing unless GH_HISTORY_TOKEN is set."""

    def __init__(self, token: str | None = None, repo: str | None = None):
        self.token = (token or os.environ.get("GH_HISTORY_TOKEN", "")).strip()
        self.repo = repo or os.environ.get("GH_HISTORY_REPO", "NANDAN-CREATOR/pricewatch")
        self._seen: dict[str, float] = {}
        self._day = ""
        self._n = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def _allowed(self, rec: dict) -> bool:
        now, today = time.time(), datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if self._day != today:
                self._day, self._n = today, 0
            if self._n >= MAX_PUSHES_PER_DAY or now - self._seen.get(rec["id"], 0) < PUSH_COOLDOWN_S:
                return False
            self._seen[rec["id"]] = now
            self._n += 1
            return True

    def _put(self, rec: dict, prefix: str = "") -> None:
        path = f"data/inbox/{prefix}{rec['ts'].replace(':', '')}-{rec['id'].replace('~', '_')[:60]}.json"
        body = json.dumps({"message": f"data: search {rec['id'][:40]}", "branch": "main",
                           "content": base64.b64encode(json.dumps(rec, ensure_ascii=False).encode()).decode()}).encode()
        req = urllib.request.Request(f"https://api.github.com/repos/{self.repo}/contents/{path}", data=body, method="PUT",
                                     headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                              "Content-Type": "application/json", "User-Agent": "pricewatch-space"})
        urllib.request.urlopen(req, timeout=20).read()

    def save(self, rec: dict | None) -> None:
        """Fire-and-forget: a failure here must never break a search."""
        if not (self.enabled and rec and self._allowed(rec)):
            return

        def work():
            try:
                self._put(rec)
            except Exception as exc:  # noqa: BLE001
                print(f"[pricewatch] could not save search history: {exc!r}")

        threading.Thread(target=work, daemon=True).start()


def make_command(cmd: str, name: str, gender: str | None, category: str, now: datetime | None = None) -> dict:
    """A request to start (`track`) or stop (`untrack`) scheduled checks for a product."""
    if cmd not in ("track", "untrack"):
        raise ValueError("cmd must be track or untrack")
    return {"cmd": cmd, "ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "name": name.strip(),
            "gender": gender, "category": category, "id": product_id(name, gender)}


def send_command(box: "GitHubInbox", rec: dict) -> None:
    """Save a command to the repo now (not fire-and-forget): the caller reports failures to the user."""
    if not box.enabled:
        raise RuntimeError("Saving to GitHub is not set up on this Space (GH_HISTORY_TOKEN missing).")
    box._put(rec, prefix="cmd-")


def ingest(inbox_dir: Path, history: History, registry: dict, delete: bool = True) -> int:
    """Fold inbox files into history + registry. Returns how many files were used."""
    n = 0
    for f in sorted(Path(inbox_dir).glob("*.json")) if Path(inbox_dir).exists() else []:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("cmd") in ("track", "untrack"):
                p = registry.setdefault(rec["id"], {"id": rec["id"], "name": rec["name"], "searches": 0,
                                                    "category": rec.get("category", "general"), "gender": rec.get("gender")})
                p["tracked"] = rec["cmd"] == "track"
                if delete:
                    f.unlink()
                continue
            offers = [Offer(store=s["store"], title=rec["name"], price=float(s["price"]), url=s.get("url", ""))
                      for s in rec["stores"]]
            ts = datetime.fromisoformat(rec["ts"])
            pid = rec["id"]
        except (ValueError, KeyError, TypeError):
            if delete:
                f.unlink()
            continue
        history.record(pid, offers, ts=ts)
        p = registry.setdefault(pid, {"id": pid, "name": rec["name"], "category": rec.get("category", "general"),
                                      "gender": rec.get("gender"), "searches": 0})
        p["searches"] = p.get("searches", 0) + 1
        p["last_seen"] = max(p.get("last_seen", ""), rec["ts"])
        p["stores"] = rec["stores"]
        n += 1
        if delete:
            f.unlink()
    return n
