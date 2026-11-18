"""Polite HTTP: identify ourselves, obey robots.txt, and wait between requests."""

from __future__ import annotations

import time
import urllib.request
import urllib.robotparser
from urllib.parse import urlparse

USER_AGENT = "pricewatch/0.1 (+https://github.com/NANDAN-CREATOR/pricewatch; polite price comparison)"


class PoliteFetcher:
    def __init__(self, delay_seconds: float = 2.0, timeout: float = 15.0):
        self.delay = delay_seconds
        self.timeout = timeout
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_request = 0.0

    def allowed(self, url: str) -> bool:
        """True only if robots.txt was read and permits this URL. If it can't be read we stay out."""
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
                self._robots[base] = rp
            except Exception:
                self._robots[base] = None
        rp = self._robots[base]
        return rp.can_fetch(USER_AGENT, url) if rp else False

    def get_bytes(self, url: str, accept: str = "*/*", max_bytes: int = 5_000_000) -> bytes:
        if not self.allowed(url):
            raise PermissionError("blocked by robots.txt (or robots.txt unreachable)")
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read(max_bytes + 1)
        finally:
            self._last_request = time.monotonic()
        if len(data) > max_bytes:
            raise ValueError(f"response larger than {max_bytes} bytes")
        return data

    def get(self, url: str, accept: str = "text/html") -> str:
        return self.get_bytes(url, accept).decode("utf-8", errors="replace")
