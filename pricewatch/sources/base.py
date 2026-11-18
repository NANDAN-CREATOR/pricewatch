from __future__ import annotations

from abc import ABC, abstractmethod

from ..match import Product
from ..models import Offer


class Source(ABC):
    """A way of getting offers for a product. One broken source must never break the run."""

    name: str = "source"
    warnings: list[str] = []  # per-store problems from the last search() that did not abort it

    @abstractmethod
    def search(self, product: Product) -> list[Offer]:
        """Return raw (unfiltered) offers. Raise on failure; the runner records the error."""
