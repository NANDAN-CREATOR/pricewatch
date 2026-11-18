# pricewatch

Find the **lowest price**, the **median/average price**, and how much each store **marks it up**, for any product you put on a watchlist. Free to run: GitHub Actions collects prices on a schedule and GitHub Pages serves the site. **No AI, no paid services, no dependencies** (Python standard library only).

## What the numbers mean

| Metric | Definition |
|---|---|
| Lowest | Cheapest in-stock offer across stores |
| Markup vs lowest | `(store price - lowest) / lowest` |
| Markup vs median | `(store price - median) / median` |
| Claimed discount | `(MRP - price) / MRP`, what the store advertises |
| MRP inflation | `(MRP - reference) / reference`, how far the "original price" sits above the real going price. `reference` = median selling price over the last 90 days across stores (today's median until history builds up) |

Outliers (outside 0.5x to 2x the median, with 4+ stores) are excluded from the stats and shown separately. Accessories, refurbished items and other variants are rejected by explicit rules and listed on the page with the reason.

## Quick start

```bash
pip install -r requirements.txt && playwright install chromium
python -m pytest -q                      # 46 tests
python -m pricewatch --source sample     # offline demo -> site/site.json
cd site && python -m http.server 8000    # open http://localhost:8000
```

## Track a real product

Edit `watchlist.json`:

```json
{
  "id": "sony-wh1000xm5",
  "name": "Sony WH-1000XM5",
  "must_include": ["sony", "wh1000xm5"],
  "must_exclude": ["xm4"],
  "min_price": 10000,
  "urls": {"StoreA": "https://...", "StoreB": "https://..."}
}
```

- `must_include`: every entry must appear in the listing title. Use `"a|b"` for alternatives. Model hyphens and units are normalised (`WH-1000XM5` = `wh1000xm5`, `128 GB` = `128gb`).
- `must_exclude`: extra words that disqualify a listing (case, cover, refurbished... are excluded by default).
- `urls`: product pages to read. The `jsonld` source reads schema.org prices that most stores publish for SEO.

## Deploy (GitHub)

1. Push to `github.com/NANDAN-CREATOR/pricewatch`.
2. Settings -> Pages -> Source: **GitHub Actions**.
3. Settings -> Actions -> General -> Workflow permissions: **Read and write**.
4. Actions -> "Update prices and deploy site" -> Run workflow (tick *demo* the first time).

It then runs every 6 hours, commits `data/history.jsonl`, and redeploys the site.

## Clothing

Set `"category": "clothing"` and `"gender": "men" | "women"` on a product. Listings for the other gender or for kids are rejected automatically, and `must_exclude` handles bundles (`"pack"`, `"combo"`). Match on the model/style, not the colour or size: variants share a price on most stores, and the Shopify reader picks the cheapest in-stock size.

```json
{
  "id": "levis-511-slim-jeans-men",
  "name": "Levi's 511 Slim Fit Jeans (Men)",
  "category": "clothing", "gender": "men",
  "must_include": ["levis|levi", "511", "jeans"],
  "must_exclude": ["pack", "combo"],
  "min_price": 1000,
  "shopify": {"BrandStore": "https://brand.example/products/511-slim-fit-jeans"},
  "urls": {"OtherStore": "https://other.example/p/12345"}
}
```

- `shopify`: product pages on Shopify stores. Reads `/products/<handle>.json`, which includes `compare_at_price` (the struck-through MRP), so **MRP inflation is exact**. Check a store by opening `https://<store>/products/<handle>.json` in a browser first.
- `urls`: any other store whose product page publishes schema.org JSON-LD (the `jsonld` source).

## Automated scraping (browser source)

`pages` + the `browser` source open each product page in a headless Chromium (Playwright), wait for JavaScript to render, and read the price. Per page it tries your CSS selectors first, then schema.org JSON-LD, then OpenGraph tags, so many stores work with no selectors at all.

```json
"pages": {"StoreA": "https://storea.example/p/511-jeans", "StoreB": "https://storeb.example/item/998"},
"source_options": {"browser": {"delay_seconds": 5, "stores": {
  "StoreA": {"title": "h1", "price": ".final-price", "mrp": ".strike-price", "image": "img.main", "sold_out_text": "Sold out"}}}}
```

To find a store's selectors: open a product page, right-click the price -> Inspect, and copy a stable class or id.

It identifies itself honestly, obeys robots.txt, waits between pages, and does **not** hide itself, rotate proxies or solve captchas. Pages that block it report an error.

## Image check ("is it the same product?")

Add `"image_check": {"mode": "flag", "threshold": 10}` to the watchlist. Each store's product image is fingerprinted (difference hash, Pillow only, no AI model) and compared with the consensus of the other stores. Needs 3+ images. `flag` marks outliers on the page ("image differs"); `reject` removes them from the statistics.

Good at: the same photo re-sized or re-hosted (brands usually supply one photo to every store). Weak at: different photos of the same item, or near-identical white-background packshots. That is why the default is `flag`, not `reject`. A stronger CLIP-style embedding model can replace `dhash` later behind `check_images()`.

## Limits (read this)

- The `jsonld` source respects `robots.txt`, rate-limits itself, and does not bypass bot protection. Big marketplaces often block GitHub's datacenter IPs and forbid scraping in their terms; those stores will show as a source error, not break the run. Prefer stores that publish JSON-LD, or add an adapter for an official API (see `pricewatch/sources/base.py`).
- A static site can't search arbitrary products live. Products are the ones in the watchlist.
- Cars: on-road price depends on city and variant, so treat them as a separate category later.

## Layout

```
pricewatch/normalize.py   price + title normalisation
pricewatch/match.py       rule-based product matching
pricewatch/stats.py       lowest / median / markup / MRP inflation / outliers
pricewatch/store.py       append-only history (data/history.jsonl)
pricewatch/sources/       sample (offline), jsonld (product pages); add more here
pricewatch/runner.py      one full run
site/index.html           static UI reading site.json
.github/workflows/        ci.yml (tests), update.yml (schedule + Pages deploy)
```

## Later: where AI can plug in

Matching is rule-based today (`match.py`). The natural upgrade is an embedding + LLM verifier for ambiguous listings behind the same `filter_offers` interface, evaluated against a labelled set.
