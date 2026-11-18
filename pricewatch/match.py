"""Rule-based product matching (no AI): decide if a store listing is the product we track."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import Offer
from .normalize import token_set, tokenize

# Words that almost always mean "not the product itself".
DEFAULT_EXCLUDE = frozenset(
    {
        "case", "cover", "protector", "guard", "skin", "strap", "sticker",
        "refurbished", "renewed", "used", "preowned", "secondhand",
        "compatible", "replacement", "dummy", "toy", "tempered", "glass",
    }
)


# Clothing: a listing for the wrong gender or for kids is a different product.
KIDS_WORDS = frozenset({"kids", "kid", "boys", "boy", "girls", "girl", "baby", "infant", "toddler", "junior"})
FEMALE_WORDS = frozenset({"women", "woman", "womens", "ladies", "lady", "female"})
MALE_WORDS = frozenset({"men", "man", "mens", "male", "gents"})


@dataclass
class Product:
    id: str
    name: str
    # Each entry is a token or "a|b" alternatives. Default: every token of `name`.
    must_include: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    urls: dict[str, str] = field(default_factory=dict)  # store -> product page (jsonld source)
    shopify: dict[str, str] = field(default_factory=dict)  # store -> product page (shopify source)
    pages: dict[str, str] = field(default_factory=dict)  # store -> product page (browser source)
    category: str = "general"  # "clothing" turns on the gender / kids rules
    gender: str | None = None  # "men" | "women" | "unisex" (clothing)
    soft_share: float = 1.0  # <1: for free-text queries, tokens WITHOUT digits only need this share to match

    @classmethod
    def from_dict(cls, d: dict) -> "Product":
        return cls(
            id=d["id"],
            name=d["name"],
            must_include=list(d.get("must_include", [])),
            must_exclude=list(d.get("must_exclude", [])),
            min_price=d.get("min_price"),
            max_price=d.get("max_price"),
            urls=dict(d.get("urls", {})),
            shopify=dict(d.get("shopify", {})),
            pages=dict(d.get("pages", {})),
            category=d.get("category", "general"),
            gender=d.get("gender"),
        )

    def required_groups(self) -> list[set[str]]:
        if not self.must_include:
            return [{t} for t in tokenize(self.name)]
        return [{t for alt in entry.split("|") for t in tokenize(alt)} for entry in self.must_include]

    def excluded(self) -> set[str]:
        out = set(DEFAULT_EXCLUDE) | {e.lower() for e in self.must_exclude}
        if self.category == "clothing":
            if self.gender != "kids":
                out |= KIDS_WORDS
            if self.gender == "men":
                out |= FEMALE_WORDS
            elif self.gender == "women":
                out |= MALE_WORDS
        return out


def explain(product: Product, offer: Offer) -> str | None:
    """Return None if the offer matches, else a short reason it was rejected."""
    title_tokens = token_set(offer.title)

    bad = product.excluded() & title_tokens
    if bad:
        return f"excluded word: {sorted(bad)[0]}"

    groups = product.required_groups()
    if product.soft_share < 1.0:
        # model numbers / sizes (tokens with digits) are mandatory; plain words only need a share
        hard = [g for g in groups if any(c.isdigit() for t in g for c in t)]
        soft = [g for g in groups if g not in hard]
    else:
        hard, soft = groups, []
    for group in hard:
        if not (group & title_tokens):
            return f"missing: {'|'.join(sorted(group))}"
    if soft:
        need = math.ceil(product.soft_share * len(soft))
        have = sum(1 for g in soft if g & title_tokens)
        if have < need:
            missing = [sorted(g)[0] for g in soft if not (g & title_tokens)]
            return f"missing: {', '.join(missing)}"

    if product.min_price is not None and offer.price < product.min_price:
        return f"price below min ({offer.price:g} < {product.min_price:g})"
    if product.max_price is not None and offer.price > product.max_price:
        return f"price above max ({offer.price:g} > {product.max_price:g})"
    return None


def filter_offers(product: Product, offers: list[Offer]) -> tuple[list[Offer], list[dict]]:
    """Split offers into (accepted, rejected-with-reason)."""
    accepted: list[Offer] = []
    rejected: list[dict] = []
    for offer in offers:
        reason = explain(product, offer)
        if reason is None:
            accepted.append(offer)
        else:
            rejected.append({"store": offer.store, "title": offer.title, "reason": reason})
    return accepted, rejected
