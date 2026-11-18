"""The whole job for one request: name and/or photo -> listings -> same-product check -> price stats."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image

from .match import Product, filter_offers
from .models import Offer
from .search import SearchProvider
from .stats import best_offer_per_store, summarize
from .vision import Embedder, consensus_similarity, cosine, describe_photo, fetch_images

PHOTO_FLOOR = 0.50  # photo search: never keep a listing less similar than this
PHOTO_MARGIN = 0.12  # ...and keep only listings within this of the best visual match
CONSENSUS_FLAG_BELOW = 0.80  # name-only search: flag listings that look unlike the others


@dataclass
class Request:
    query: str = ""
    photo: Image.Image | None = None
    category: str = "general"  # "clothing" turns on gender / kids rules
    gender: str | None = None  # "men" | "women" | None
    min_similarity: float | None = None  # photo search: fixed cut-off; None = automatic (best match minus a margin)


def find(req: Request, provider: SearchProvider, embedder: Embedder | None, offers_cache=None) -> dict:
    """Run one search. Never raises for image problems: it notes them and carries on without images."""
    query = (req.query or "").strip()
    if not query and req.photo is None:
        raise ValueError("Type a product name or upload a photo.")
    notes: list[str] = []
    photo_vec = None
    photo_info = None

    if req.photo is not None:
        if embedder is None:
            notes.append("The image model is not available, so the photo was not used.")
        else:
            try:
                photo_vec = embedder.embed_images([req.photo])[0]
                if not query:
                    photo_info = describe_photo(req.photo, embedder, req.gender)
                    query = photo_info["query"]
            except Exception as exc:  # model/GPU hiccup: fall back to text only
                notes.append(f"The image model failed ({type(exc).__name__}: {str(exc)[:160]}), so the photo was not used.")
    if not query:
        why = (" " + " ".join(notes)) if notes else ""
        raise ValueError("I could not work out a search phrase from the photo. Please type a product name too." + why)

    fetched = offers_cache(query) if offers_cache else provider.search(query)
    # a cache may return (offers, fetched_at_iso) so the page shows when the prices were really read
    raw, checked_at = fetched if isinstance(fetched, tuple) else (fetched, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    product = Product(id="adhoc", name=query, category=req.category, gender=req.gender, soft_share=0.5)
    accepted, rejected = filter_offers(product, raw)
    similarity: dict[str, float] = {}
    cutoff = None
    if photo_vec is not None and embedder is not None:
        # Photo search: compare every plausible listing with the photo, keep the best visual match per store.
        try:
            cands = accepted[:30]
            images = fetch_images([o.image_url for o in cands])
            have = [(o, im) for o, im in zip(cands, images) if im is not None]
            sims = [float(cosine(photo_vec, v)) for v in embedder.embed_images([im for _, im in have])] if have else []
            if sims:
                cutoff = req.min_similarity if req.min_similarity else max(PHOTO_FLOOR, max(sims) - PHOTO_MARGIN)
                by_store: dict[str, tuple[float, Offer]] = {}
                for (o, _), sim in zip(have, sims):
                    if sim >= cutoff:
                        cur = by_store.get(o.store)
                        if cur is None or (sim, -o.price) > (cur[0], -cur[1].price):
                            by_store[o.store] = (sim, o)
                    else:
                        rejected.append({"store": o.store, "title": o.title,
                                         "reason": f"looks different from your photo ({sim:.0%} similar)"})
                for o in cands:
                    if all(o is not h for h, _ in have):
                        rejected.append({"store": o.store, "title": o.title, "reason": "no picture to compare with your photo"})
                accepted = [o for _, o in by_store.values()]
                similarity = {st: round(sim, 3) for st, (sim, _) in by_store.items()}
            else:
                notes.append("No listing pictures could be loaded, so results are not matched to your photo.")
        except Exception as exc:
            notes.append(f"Image comparison failed ({type(exc).__name__}); showing results without it.")
    elif embedder is not None:
        per_store = best_offer_per_store(accepted)
        if len(per_store) >= 3:  # name-only: flag listings whose picture differs from the group's
            try:
                images = fetch_images([o.image_url for o in per_store])
                have = [(o, im) for o, im in zip(per_store, images) if im is not None]
                if len(have) >= 3:
                    sims = consensus_similarity(embedder.embed_images([im for _, im in have]))
                    similarity = {o.store: round(float(sv), 3) for (o, _), sv in zip(have, sims)}
            except Exception as exc:
                notes.append(f"Image comparison failed ({type(exc).__name__}); showing results without it.")
    flag_below = cutoff if cutoff is not None else CONSENSUS_FLAG_BELOW

    summary = summarize(accepted)
    if summary:
        for row in summary["stores"]:
            sim = similarity.get(row["store"])
            row["similarity"] = sim
            row["looks_same"] = None if sim is None else sim >= flag_below
            row["checked_at"] = checked_at
    return {"query": query, "photo_info": photo_info, "summary": summary, "rejected": rejected,
            "notes": notes, "listings_found": len(raw), "checked_at": checked_at,
            "photo_used": cutoff is not None}
