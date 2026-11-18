"""Turn messy store text into comparable tokens and numbers."""

from __future__ import annotations

import re

_PRICE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(gb|tb|mb|mp|inch|in|hz|mah|kg|ml|w|g|l)\b")
_INTRA_HYPHEN_RE = re.compile(r"(?<=\w)-(?=\w)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9.]+")

STOPWORDS = frozenset(
    {
        "with", "for", "and", "the", "of", "in", "by", "new", "latest",
        "buy", "online", "india", "official", "genuine", "original", "free",
    }
)


def parse_price(value: str | int | float | None) -> float | None:
    """Parse '₹1,29,999.00', 'Rs. 499', 'INR 1,299' or a number into a float.

    Returns None when no usable positive number is found.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    match = _PRICE_RE.search(str(value))
    if not match:
        return None
    number = float(match.group(0).replace(",", ""))
    return number if number > 0 else None


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; 'WH-1000XM5' -> 'wh1000xm5', '128 GB' -> '128gb'."""
    text = text.lower()
    text = _INTRA_HYPHEN_RE.sub("", text)
    text = _UNIT_RE.sub(lambda m: m.group(1) + m.group(2), text)
    tokens = [t.strip(".") for t in _NON_ALNUM_RE.split(text)]
    return [t for t in tokens if t and t not in STOPWORDS]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))
