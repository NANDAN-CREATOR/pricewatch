from __future__ import annotations

from typing import Protocol

from ..models import Offer


class SearchError(RuntimeError):
    """The search provider failed or is not configured. The message is safe to show to the user."""


class SearchAuthError(SearchError):
    """The provider rejected the API key (so another provider may be the right one for this key)."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str) -> list[Offer]:
        """Return raw listings (unfiltered) for a free-text product query."""


class FirstAccepted:
    """Try each provider in turn; move on only when one rejects the API key (wrong service for that key)."""

    def __init__(self, providers):
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self._good = None

    def search(self, query):
        order = ([self._good] if self._good else []) + [p for p in self.providers if p is not self._good]
        last = None
        for p in order:
            try:
                out = p.search(query)
                self._good = p
                return out
            except SearchAuthError as exc:
                last = exc
        raise SearchError("The search API key was rejected by both SerpApi and SearchApi. "
                          "Check the key in the Space secret.") from last
