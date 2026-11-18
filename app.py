"""Gradio app: type a product name and/or upload a photo -> lowest price, median, and store markup %.

Runs as a Hugging Face Space (ZeroGPU) or locally:  pip install -r requirements-space.txt gradio; python app.py
Needs the SERPAPI_KEY secret for store search.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

import gradio as gr

from pricewatch.finder import PHOTO_MIN_SIMILARITY, Request, find
from pricewatch.search import COUNTRIES, FirstAccepted, SearchError, clean_country
from pricewatch.search.searchapi import SearchApiShopping
from pricewatch.search.serpapi import SerpApiShopping
from pricewatch.vision import ClipEmbedder

MAX_SEARCHES_PER_DAY = int(os.environ.get("MAX_SEARCHES_PER_DAY", "8"))  # protects the free search quota
CACHE_SECONDS = 12 * 3600

PROVIDER = FirstAccepted([SearchApiShopping(), SerpApiShopping()])  # whichever service the key belongs to
if os.environ.get("SPACE_ID"):
    try:  # a ZeroGPU Space refuses to start unless one @spaces.GPU function exists; ours is never called
        import spaces

        @spaces.GPU
        def _gpu_placeholder():
            return None
    except ImportError:
        pass

try:
    EMBEDDER = ClipEmbedder()
    if os.environ.get("SPACE_ID"):
        EMBEDDER.load()  # load weights once at startup on the Space
except Exception as exc:  # the app still works for name-only price search
    print(f"[pricewatch] image model unavailable: {exc!r}")
    EMBEDDER = None

_lock = threading.Lock()
_cache: dict[str, tuple[float, list, str]] = {}
_budget = {"day": "", "used": 0}


def cached_search(query: str, country: str = "in"):
    """Search with a 12h cache and a daily cap, so repeated questions do not burn the free quota."""
    key = clean_country(country) + "|" + " ".join(query.lower().split())
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1], hit[2]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if _budget["day"] != today:
            _budget.update(day=today, used=0)
        if _budget["used"] >= MAX_SEARCHES_PER_DAY:
            raise SearchError("The free daily search limit for this app is used up. Try again tomorrow.")
        _budget["used"] += 1
    try:
        offers = PROVIDER.search(query, country)
    except Exception:
        with _lock:
            _budget["used"] -= 1  # a failed call should not use up the allowance
        raise
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
    with _lock:
        _cache[key] = (now, offers, stamp)
    return offers, stamp


SYMBOLS = {"INR": "₹", "USD": "$", "GBP": "£", "CAD": "CA$", "AUD": "A$", "AED": "AED ", "SGD": "S$", "EUR": "€"}


def money(n: float, currency: str = "INR") -> str:
    """₹12,34,567 for rupees (Indian digit grouping); $1,234,567 style for the rest."""
    sym = SYMBOLS.get(currency, currency + " ")
    if currency != "INR":
        return f"{sym}{n:,.0f}"
    s = f"{int(round(n))}"
    head, tail = s[:-3], s[-3:]
    if head:
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return sym + s


def pct(v) -> str:
    return "-" if v is None else f"{v:+.1f}%"


def search(query, photo, kind, gender, min_similarity, country="in"):
    req = Request(
        query=query or "",
        photo=photo,
        category="clothing" if kind == "Clothing & shoes" else "general",
        gender={"Men": "men", "Women": "women"}.get(gender),
        min_similarity=min_similarity,
        country=country,
    )
    try:
        out = find(req, PROVIDER, EMBEDDER, offers_cache=cached_search)
    except (ValueError, SearchError) as exc:
        return f"**{exc}**", [], ""

    lines = [f"**Searched for:** {out['query']}"]
    if out["photo_info"]:
        lines.append(f"*(guessed from your photo: {out['photo_info']['color']} {out['photo_info']['category']}. "
                     "Add the brand and model in the box for a sharper result.)*")
    s = out["summary"]
    cur = s["currency"] if s else "INR"
    inr = lambda n: money(n, cur)  # noqa: E731
    if not s:
        lines.append(f"No matching listings. {out['listings_found']} raw listings were found and all were filtered out "
                     "(wrong product, wrong gender, accessories or used items). Try a more general name.")
    else:
        lines += [
            f"## Lowest: {inr(s['lowest']['price'])} at {s['lowest']['store']}",
            f"Median {inr(s['median'])} · average {inr(s['average'])} · {s['store_count']} stores compared",
            "*MRP inflation compares each store's struck-through price with today's median price across stores.*",
        ]
    lines += [f"\n> {n}" for n in out["notes"]]

    rows = []
    for r in (s or {}).get("stores", []):
        sim = r.get("similarity")
        same = {True: "yes", False: "no"}.get(r.get("looks_same"), "-")
        rows.append([
            r["store"], inr(r["price"]), pct(r["markup_vs_lowest_pct"]), pct(r["markup_vs_median_pct"]),
            "-" if r["claimed_discount_pct"] is None else f"{r['claimed_discount_pct']:.0f}% off",
            pct(r["mrp_inflation_pct"]), "-" if sim is None else f"{sim:.0%}", same,
            f"[open]({r['url']})" if r["url"] else "-",
        ])

    rejected = ""
    if out["rejected"]:
        rejected = "\n".join(f"- {x['store']}: {x['title']} → {x['reason']}" for x in out["rejected"][:40])
        rejected = f"### Filtered out ({len(out['rejected'])})\n{rejected}"
    if s and s["outliers"]:
        rejected += "\n\n### Price outliers excluded\n" + "\n".join(
            f"- {o['store']}: {inr(o['price'])} · {o['title']}" for o in s["outliers"])
    return "\n\n".join(lines), rows, rejected


def api_find(query, photo, kind, gender, min_similarity, country="in"):
    """JSON endpoint used by the GitHub Pages site (api_name="find")."""
    req = Request(query=query or "", photo=photo,
                  category="clothing" if kind == "Clothing & shoes" else "general",
                  gender={"Men": "men", "Women": "women"}.get(gender),
                  min_similarity=min_similarity, country=country)
    try:
        return {"ok": True, **find(req, PROVIDER, EMBEDDER, offers_cache=cached_search)}
    except (ValueError, SearchError) as exc:
        return {"ok": False, "error": str(exc)}


with gr.Blocks(title="pricewatch") as demo:
    gr.Markdown("# pricewatch\nType a product name, upload a photo, or both. "
                "I find the lowest price across stores and show how much each one marks it up.")
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Product name", placeholder="e.g. Levi's 511 slim fit jeans")
            photo = gr.Image(label="Or upload a photo", type="pil", height=260)
            with gr.Row():
                kind = gr.Radio(["Anything", "Clothing & shoes"], value="Anything", label="Type")
                gender = gr.Radio(["Any", "Men", "Women"], value="Any", label="Gender (clothing)")
            country = gr.Dropdown([(n, c) for c, (n, _) in COUNTRIES.items()], value="in", label="Country")
            min_sim = gr.Slider(0.5, 0.95, value=PHOTO_MIN_SIMILARITY, step=0.01,
                                label="Photo match strictness (higher = only very similar items)")
            go = gr.Button("Find prices", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
    table = gr.Dataframe(
        headers=["Store", "Price", "Markup vs lowest", "Markup vs median", "Claims", "MRP inflation",
                 "Image similarity", "Looks same?", "Link"],
        datatype=["str"] * 8 + ["markdown"], interactive=False, wrap=True)
    filtered = gr.Markdown()
    go.click(search, [query, photo, kind, gender, min_sim, country], [summary, table, filtered])
    query.submit(search, [query, photo, kind, gender, min_sim, country], [summary, table, filtered])
    _j = gr.JSON(visible=False)
    gr.Button(visible=False).click(api_find, [query, photo, kind, gender, min_sim, country], _j, api_name="find")

if __name__ == "__main__":
    demo.launch()
