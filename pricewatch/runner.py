"""Orchestrates one full run: fetch -> match -> (image check) -> stats -> history -> site data."""

from __future__ import annotations

import json
from pathlib import Path

from .imagematch import DEFAULT_THRESHOLD, check_images
from .match import Product, filter_offers
from .models import Offer
from .polite import PoliteFetcher
from .sources import build
from .stats import best_offer_per_store, summarize
from .store import History, now_utc


def _image_pass(offers: list[Offer], cfg: dict, fetcher: PoliteFetcher) -> dict[str, dict]:
    """Download each store's product image and compare them. Returns store -> {distance, match}."""
    images: dict[str, bytes] = {}
    for o in offers:
        if o.image_url:
            try:
                images[o.store] = fetcher.get_bytes(o.image_url, accept="image/*")
            except Exception:
                pass  # no image, no verdict; never fail the run over a picture
    return check_images(images, cfg.get("threshold", DEFAULT_THRESHOLD))


def run(config: dict, history: History, source_overrides: list[str] | None = None) -> dict:
    products = [Product.from_dict(p) for p in config["products"]]
    source_names = source_overrides or config.get("sources", ["jsonld"])
    sources = [build(n, **config.get("source_options", {}).get(n, {})) for n in source_names]

    image_cfg = config.get("image_check", {})  # {"mode": "flag" | "reject", "threshold": 10}
    image_mode = image_cfg.get("mode")  # None = off
    image_fetcher = PoliteFetcher(image_cfg.get("delay_seconds", 1.0)) if image_mode else None

    results = []
    for product in products:
        raw, errors = [], []
        for source in sources:
            try:
                raw.extend(source.search(product))
                errors.extend({"source": source.name, "error": w} for w in source.warnings)
            except Exception as exc:  # one broken source must not kill the run
                errors.append({"source": source.name, "error": str(exc)})

        accepted, rejected = filter_offers(product, raw)

        image_results: dict[str, dict] = {}
        if image_mode:
            per_store = best_offer_per_store(accepted)
            image_results = _image_pass(per_store, image_cfg, image_fetcher)
            if image_mode == "reject":
                bad = {s for s, r in image_results.items() if r["match"] is False}
                for o in per_store:
                    if o.store in bad:
                        rejected.append({"store": o.store, "title": o.title,
                                         "reason": f"image differs from the other stores (distance {image_results[o.store]['distance']})"})
                accepted = [o for o in accepted if o.store not in bad]

        reference = history.reference_price(product.id)  # computed BEFORE adding today's prices
        summary = summarize(accepted, reference_price=reference)
        if summary:
            for row in summary["stores"]:
                row["image_match"] = image_results.get(row["store"], {}).get("match")
                row["image_distance"] = image_results.get(row["store"], {}).get("distance")

        entry = {
            "id": product.id,
            "name": product.name,
            "category": product.category,
            "errors": errors,
            "rejected": rejected,
            "summary": summary,
            "history": [],
        }
        if summary:
            kept_stores = {s["store"] for s in summary["stores"]}
            history.record(product.id, [o for o in accepted if o.store in kept_stores])
            entry["history"] = history.series(product.id)
        results.append(entry)

    return {"generated_at": now_utc().isoformat(timespec="seconds"), "products": results}


def write_site_data(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
