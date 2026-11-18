from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Offer:
    """One store's listing for one product at one moment."""

    store: str
    title: str
    price: float
    url: str = ""
    mrp: float | None = None  # the "original" / struck-through price the store shows
    currency: str = "INR"
    in_stock: bool = True
    image_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
