from __future__ import annotations

from typing import Protocol

from ..models import Offer


class SearchError(RuntimeError):
    """The search provider failed or is not configured. The message is safe to show to the user."""


class SearchAuthError(SearchError):
    """The provider rejected the API key (so another provider may be the right one for this key)."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, country: str = "in") -> list[Offer]:
        """Return raw listings (unfiltered) for a free-text product query in one country's market."""


class FirstAccepted:
    """Try each provider in turn; move on only when one rejects the API key (wrong service for that key)."""

    def __init__(self, providers):
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self._good = None

    def search(self, query, country="in"):
        order = ([self._good] if self._good else []) + [p for p in self.providers if p is not self._good]
        last = None
        for p in order:
            try:
                out = p.search(query, country)
                self._good = p
                return out
            except SearchAuthError as exc:
                last = exc
        raise SearchError("The search API key was rejected by both SerpApi and SearchApi. "
                          "Check the key in the Space secret.") from last


# Countries offered in the app: code -> (name, ISO currency). Codes are Google's `gl` values.
COUNTRIES = {
    "in": ("India", "INR"), "us": ("United States", "USD"), "uk": ("United Kingdom", "GBP"),
    "ca": ("Canada", "CAD"), "au": ("Australia", "AUD"), "ae": ("United Arab Emirates", "AED"),
    "sg": ("Singapore", "SGD"), "de": ("Germany", "EUR"), "fr": ("France", "EUR"),
}


def currency_for(country: str) -> str:
    return COUNTRIES.get(country, ("", "USD"))[1]


def clean_country(country: str | None) -> str:
    c = (country or "in").strip().lower()
    return c if c in COUNTRIES else "in"
