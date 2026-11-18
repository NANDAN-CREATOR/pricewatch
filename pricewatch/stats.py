"""Price statistics: lowest, average, and how much each store inflates the price."""

from __future__ import annotations

import statistics

from .models import Offer

# An offer priced outside [median * LOW, median * HIGH] is treated as an outlier
# (wrong variant, bundle, or a data error) and excluded from the statistics.
OUTLIER_LOW = 0.5
OUTLIER_HIGH = 2.0
MIN_OFFERS_FOR_OUTLIERS = 4


def best_offer_per_store(offers: list[Offer]) -> list[Offer]:
    """Keep only the cheapest in-stock listing for each store (falls back to out-of-stock)."""
    best: dict[str, Offer] = {}
    for offer in offers:
        current = best.get(offer.store)
        if current is None or _rank(offer) < _rank(current):
            best[offer.store] = offer
    return list(best.values())


def _rank(offer: Offer) -> tuple[int, float]:
    return (0 if offer.in_stock else 1, offer.price)


def split_outliers(offers: list[Offer]) -> tuple[list[Offer], list[Offer]]:
    if len(offers) < MIN_OFFERS_FOR_OUTLIERS:
        return offers, []
    median = statistics.median(o.price for o in offers)
    kept, dropped = [], []
    for offer in offers:
        (kept if OUTLIER_LOW * median <= offer.price <= OUTLIER_HIGH * median else dropped).append(offer)
    return kept, dropped


def pct(part: float, base: float) -> float:
    return round((part - base) / base * 100, 1)


def summarize(offers: list[Offer], reference_price: float | None = None) -> dict | None:
    """Build the comparison for one product.

    reference_price: the product's typical historical selling price (e.g. 90-day median).
    It is used to judge whether a store's struck-through MRP is inflated.
    Returns None when there are no offers.
    """
    per_store = best_offer_per_store(offers)
    kept, dropped = split_outliers(per_store)
    if not kept:
        return None

    prices = [o.price for o in kept]
    lowest = min(kept, key=lambda o: o.price)
    median = statistics.median(prices)
    mean = statistics.fmean(prices)
    ref = reference_price or median

    stores = []
    for o in sorted(kept, key=lambda o: o.price):
        row = {
            "store": o.store,
            "price": o.price,
            "url": o.url,
            "in_stock": o.in_stock,
            "markup_vs_lowest_pct": pct(o.price, lowest.price),
            "markup_vs_median_pct": pct(o.price, median),
            "mrp": o.mrp,
            "claimed_discount_pct": None,
            "mrp_inflation_pct": None,
        }
        if o.mrp and o.mrp > 0:
            # What the store says you save, e.g. MRP 1000, price 700 -> 30%.
            row["claimed_discount_pct"] = round((o.mrp - o.price) / o.mrp * 100, 1)
            # How far the MRP sits above the real going price. 0 = honest MRP.
            row["mrp_inflation_pct"] = pct(o.mrp, ref)
        stores.append(row)

    return {
        "currency": lowest.currency,
        "lowest": {"store": lowest.store, "price": lowest.price, "url": lowest.url},
        "median": round(median, 2),
        "average": round(mean, 2),
        "reference_price": round(ref, 2),
        "store_count": len(kept),
        "stores": stores,
        "outliers": [{"store": o.store, "price": o.price, "title": o.title} for o in dropped],
    }
