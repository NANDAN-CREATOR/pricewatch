from __future__ import annotations

from typing import Protocol

from ..models import Offer


class SearchError(RuntimeError):
    """The search provider failed or is not configured. The message is safe to show to the user."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str) -> list[Offer]:
        """Return raw listings (unfiltered) for a free-text product query."""
