"""The whole job for one request: name and/or photo -> listings -> same-product check -> price stats."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .match import Product, filter_offers
from .search import SearchProvider
from .stats import best_offer_per_store, summarize
from .vision import Embedder, consensus_similarity, cosine, describe_photo, fetch_images

PHOTO_MIN_SIMILARITY = 0.75  # default cut-off when comparing listings with an uploaded photo
CONSENSUS_FLAG_BELOW = 0.80  # name-only search: flag listings that look unlike the others


@dataclass
class Request:
    query: str = ""
    photo: Image.Image | None = None
    category: str = "general"  # "clothing" turns on gender / kids rules
    gender: str | None = None  # "men" | "women" | None
    min_similarity: float | None = None  # photo search: drop listings below this similarity


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
                notes.append(f"The image model failed ({type(exc).__name__}), so the photo was not used.")
    if not query:
        raise ValueError("I could not work out a search phrase from the photo. Please type a product name too.")

    raw = offers_cache(query) if offers_cache else provider.search(query)

    product = Product(id="adhoc", name=query, category=req.category, gender=req.gender, soft_share=0.6)
    accepted, rejected = filter_offers(product, raw)
    per_store = best_offer_per_store(accepted)

    similarity: dict[str, float] = {}
    if embedder is not None and per_store and (photo_vec is not None or len(per_store) >= 3):
        try:
            images = fetch_images([o.image_url for o in per_store])
            have = [(o, im) for o, im in zip(per_store, images) if im is not None]
            if have:
                vecs = embedder.embed_images([im for _, im in have])
                if photo_vec is not None:
                    sims = [cosine(photo_vec, v) for v in vecs]
                elif len(have) >= 3:
                    sims = list(consensus_similarity(vecs))
                else:
                    sims = []
                similarity = {o.store: round(float(s), 3) for (o, _), s in zip(have, sims)}
        except Exception as exc:
            notes.append(f"Image comparison failed ({type(exc).__name__}); showing results without it.")

    cutoff = None
    if photo_vec is not None:
        cutoff = req.min_similarity if req.min_similarity is not None else PHOTO_MIN_SIMILARITY
        dropped = {s for s, v in similarity.items() if v < cutoff}
        for o in per_store:
            if o.store in dropped:
                rejected.append({"store": o.store, "title": o.title,
                                 "reason": f"looks different from your photo ({similarity[o.store]:.0%} similar)"})
        accepted = [o for o in accepted if o.store not in dropped]
    flag_below = cutoff if cutoff is not None else CONSENSUS_FLAG_BELOW

    summary = summarize(accepted)
    if summary:
        for row in summary["stores"]:
            sim = similarity.get(row["store"])
            row["similarity"] = sim
            row["looks_same"] = None if sim is None else sim >= flag_below
    return {"query": query, "photo_info": photo_info, "summary": summary, "rejected": rejected,
            "notes": notes, "listings_found": len(raw)}
