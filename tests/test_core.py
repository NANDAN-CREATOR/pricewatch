import json
from datetime import timedelta
from pathlib import Path

import pytest

from pricewatch.match import Product, explain, filter_offers
from pricewatch.models import Offer
from pricewatch.normalize import parse_price, tokenize
from pricewatch.runner import run
from pricewatch.sources.jsonld import parse_jsonld
from pricewatch.stats import best_offer_per_store, summarize
from pricewatch.store import History, now_utc


# ---------- normalize ----------
@pytest.mark.parametrize(
    "raw,expected",
    [("₹1,29,999.00", 129999.0), ("Rs. 499", 499.0), ("INR 1,299", 1299.0), (2500, 2500.0),
     ("free", None), (None, None), (0, None), ("", None)],
)
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


def test_tokenize_joins_model_hyphens_and_units():
    assert "wh1000xm5" in tokenize("Sony WH-1000XM5 Headphones")
    assert "128gb" in tokenize("iPhone 15 (128 GB)")
    assert "128gb" in tokenize("iPhone 15 128GB")


# ---------- match ----------
def sony():
    return Product(id="s", name="Sony WH-1000XM5", must_include=["sony", "wh1000xm5"], min_price=10000)


def offer(title, price=25000, store="A", **kw):
    return Offer(store=store, title=title, price=price, **kw)


def test_match_accepts_real_listing_with_hyphen_variants():
    assert explain(sony(), offer("Sony WH1000XM5 Bluetooth Headphones")) is None
    assert explain(sony(), offer("Sony WH-1000XM5 Noise Cancelling")) is None


def test_match_rejects_accessory_refurb_and_other_model():
    assert "excluded" in explain(sony(), offer("Ear Pads Cover for Sony WH-1000XM5"))
    assert "excluded" in explain(sony(), offer("Sony WH-1000XM5 (Renewed)"))
    assert "missing" in explain(sony(), offer("Sony WH-1000XM4 Headphones"))


def test_match_price_bounds():
    assert "below min" in explain(sony(), offer("Sony WH-1000XM5", price=900))


def test_alternatives_with_pipe():
    p = Product(id="x", name="x", must_include=["tv|television", "55inch|55"])
    assert explain(p, offer("Samsung 55 inch Television")) is None


def test_storage_variant_must_match():
    p = Product.from_dict({"id": "i", "name": "iPhone 15", "must_include": ["iphone", "15", "128gb"],
                           "must_exclude": ["pro"]})
    assert explain(p, offer("Apple iPhone 15 (128 GB)")) is None
    assert explain(p, offer("Apple iPhone 15 256GB")) is not None
    assert explain(p, offer("Apple iPhone 15 Pro 128GB")) is not None


# ---------- stats ----------
def test_summary_numbers():
    offers = [offer("t", 100, "A", mrp=200), offer("t", 110, "B"), offer("t", 120, "C"), offer("t", 130, "D")]
    s = summarize(offers)
    assert s["lowest"] == {"store": "A", "price": 100, "url": ""}
    assert s["median"] == 115
    assert s["average"] == 115
    rows = {r["store"]: r for r in s["stores"]}
    assert rows["A"]["markup_vs_lowest_pct"] == 0
    assert rows["D"]["markup_vs_lowest_pct"] == 30.0
    assert rows["D"]["markup_vs_median_pct"] == 13.0
    assert rows["A"]["claimed_discount_pct"] == 50.0
    assert rows["A"]["mrp_inflation_pct"] == round((200 - 115) / 115 * 100, 1)


def test_reference_price_drives_mrp_inflation():
    offers = [offer("t", 700, "A", mrp=1000)]
    s = summarize(offers, reference_price=700)
    assert s["stores"][0]["mrp_inflation_pct"] == round(300 / 700 * 100, 1)


def test_outlier_dropped_from_stats():
    offers = [offer("t", p, s) for p, s in [(100, "A"), (105, "B"), (110, "C"), (1000, "D")]]
    s = summarize(offers)
    assert s["store_count"] == 3
    assert s["outliers"][0]["store"] == "D"


def test_best_offer_per_store_prefers_in_stock():
    offers = [offer("t", 90, "A", in_stock=False), offer("t", 100, "A"), offer("t", 120, "A")]
    assert best_offer_per_store(offers)[0].price == 100


def test_empty_summary_is_none():
    assert summarize([]) is None


# ---------- history ----------
def test_history_roundtrip_and_reference(tmp_path):
    h = History(tmp_path / "h.jsonl")
    h.record("p", [offer("t", 100, "A"), offer("t", 200, "B")], ts=now_utc() - timedelta(days=10))
    h.record("p", [offer("t", 50, "A")], ts=now_utc() - timedelta(days=200))  # too old
    h.save()
    h2 = History(tmp_path / "h.jsonl")
    assert h2.reference_price("p") == 150
    assert h2.reference_price("other") is None
    assert h2.series("p")[-1]["lowest"] == 100


# ---------- json-ld ----------
LD = """<html><script type="application/ld+json">
{"@context":"https://schema.org","@graph":[{"@type":"Product","name":"Sony WH-1000XM5",
 "offers":{"@type":"Offer","price":"26,990","priceCurrency":"INR","availability":"https://schema.org/InStock"}}]}
</script><script type="application/ld+json">{ not json</script></html>"""


def test_parse_jsonld_graph_and_bad_blocks():
    offers = parse_jsonld(LD, "StoreX", "https://x/p")
    assert len(offers) == 1
    o = offers[0]
    assert (o.store, o.price, o.currency, o.in_stock) == ("StoreX", 26990.0, "INR", True)


def test_parse_jsonld_out_of_stock_and_offer_list():
    html = ('<script type="application/ld+json">{"@type":"Product","name":"P","offers":['
            '{"price":10,"availability":"https://schema.org/OutOfStock"},{"lowPrice":"20"}]}</script>')
    offers = parse_jsonld(html, "S")
    assert [(o.price, o.in_stock) for o in offers] == [(10.0, False), (20.0, True)]


# ---------- end to end with the offline sample source ----------
def test_end_to_end_sample(tmp_path):
    config = json.loads(Path("watchlist.json").read_text())
    h = History(tmp_path / "h.jsonl")
    data = run(config, h, source_overrides=["sample"])
    by_id = {p["id"]: p for p in data["products"]}

    s = by_id["sony-wh1000xm5"]["summary"]
    assert s["lowest"]["store"] == "StoreB" and s["lowest"]["price"] == 24990
    assert s["store_count"] == 4
    assert len(by_id["sony-wh1000xm5"]["rejected"]) == 3

    i = by_id["iphone-15-128"]["summary"]
    assert i["lowest"]["store"] == "StoreB"
    assert [o["store"] for o in i["outliers"]] == ["StoreF"]  # 149,900 is not an iPhone 15 price
    assert i["store_count"] == 3


def test_a_failing_source_is_reported_not_fatal(tmp_path):
    config = {"products": [{"id": "x", "name": "Nothing", "urls": {"S": "https://invalid.invalid/p"}}],
              "source_options": {"jsonld": {"delay_seconds": 0}}}
    data = run(config, History(tmp_path / "h.jsonl"), source_overrides=["jsonld"])
    assert data["products"][0]["summary"] is None
    assert data["products"][0]["errors"]


# ---------- clothing ----------
from pricewatch.sources.shopify import json_url, parse_product


def jeans():
    return Product.from_dict({"id": "j", "name": "Levi's 511", "category": "clothing", "gender": "men",
                              "must_include": ["levis|levi", "511", "jeans"], "must_exclude": ["pack", "combo"]})


@pytest.mark.parametrize(
    "title,ok",
    [("Levi's Men's 511 Slim Fit Jeans", True),
     ("Levis 511 Slim Fit Men Jeans - Blue", True),
     ("Levi's Women's 511 Slim Fit Jeans", False),   # wrong gender
     ("Levi's Boys 511 Slim Jeans (Kids)", False),    # kids
     ("Levi's 501 Original Fit Jeans Men", False),    # different model
     ("Levi's 511 Slim Fit Jeans Men - Pack of 2", False)],  # bundle
)
def test_clothing_rules(title, ok):
    assert (explain(jeans(), offer(title, price=2500)) is None) is ok


def test_women_product_rejects_men_listing():
    p = Product.from_dict({"id": "w", "name": "x", "category": "clothing", "gender": "women",
                           "must_include": ["adidas"]})
    assert explain(p, offer("Adidas Women's Track Pants")) is None
    assert explain(p, offer("Adidas Track Pants Men")) is not None


def test_gender_rules_off_for_non_clothing():
    p = Product.from_dict({"id": "x", "name": "x", "must_include": ["watch"], "gender": "men"})
    assert explain(p, offer("Smart Watch for Women")) is None


SHOPIFY = {"product": {"title": "Oversized Tee", "vendor": "Acme Wear", "variants": [
    {"title": "S", "price": "799.00", "compare_at_price": "1499.00", "available": False},
    {"title": "M", "price": "849.00", "compare_at_price": "1499.00", "available": True},
    {"title": "L", "price": "899.00", "compare_at_price": None, "available": True}]}}


def test_shopify_parse_uses_cheapest_available_variant_and_compare_at_as_mrp():
    o = parse_product(SHOPIFY, "Acme", "https://acme.in/products/tee")
    assert (o.price, o.mrp, o.in_stock) == (849.0, 1499.0, True)
    assert o.title == "Acme Wear Oversized Tee"  # vendor prepended so brand matching works


def test_shopify_all_sold_out_is_out_of_stock_and_bad_compare_at_ignored():
    data = {"title": "T", "variants": [{"price": "500", "compare_at_price": "400", "available": False}]}
    o = parse_product(data, "S")
    assert o.in_stock is False and o.mrp is None  # compare_at below price is not an MRP
    assert parse_product({"title": "T", "variants": []}, "S") is None


def test_shopify_json_url():
    assert json_url("https://acme.in/collections/tees/products/oversized-tee?variant=123") == \
        "https://acme.in/products/oversized-tee.json"
    with pytest.raises(ValueError):
        json_url("https://acme.in/pages/about")


def test_end_to_end_clothing_sample(tmp_path):
    config = json.loads(Path("watchlist.json").read_text())
    data = run(config, History(tmp_path / "h.jsonl"), source_overrides=["sample"])
    by_id = {p["id"]: p for p in data["products"]}

    j = by_id["levis-511-slim-jeans-men"]
    assert j["category"] == "clothing"
    assert j["summary"]["lowest"]["store"] == "StoreB"
    assert j["summary"]["store_count"] == 4
    assert len(j["rejected"]) == 4  # women's, boys, 501, pack of 2

    t = by_id["adidas-essentials-track-pants-women"]["summary"]
    assert t["lowest"]["price"] == 2499 and t["store_count"] == 3


# ---------- images ----------
import io
import os
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw

from pricewatch.imagematch import check_images, dhash, hamming
from pricewatch.sources.jsonld import parse_meta


def make_image(kind: str, size=(400, 400), quality=90, fmt="JPEG") -> bytes:
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    w, h = size
    if kind == "jeans":  # two legs
        d.rectangle([w * .25, h * .1, w * .45, h * .9], fill=(20, 30, 90))
        d.rectangle([w * .55, h * .1, w * .75, h * .9], fill=(20, 30, 90))
        d.rectangle([w * .25, h * .1, w * .75, h * .25], fill=(10, 15, 60))
    elif kind == "tee":  # a T shape
        d.rectangle([w * .1, h * .1, w * .9, h * .35], fill=(200, 40, 40))
        d.rectangle([w * .3, h * .35, w * .7, h * .85], fill=(200, 40, 40))
    elif kind == "stripes":
        for i in range(0, int(w), 40):
            d.rectangle([i, 0, i + 20, h], fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return buf.getvalue()


def test_same_photo_resized_and_recompressed_is_close():
    a = dhash(make_image("jeans"))
    b = dhash(make_image("jeans", size=(900, 900), quality=40))
    c = dhash(make_image("jeans", size=(250, 250), fmt="PNG"))
    assert hamming(a, b) <= 6 and hamming(a, c) <= 6


def test_different_products_are_far_apart():
    assert hamming(dhash(make_image("jeans")), dhash(make_image("tee"))) > 12
    assert hamming(dhash(make_image("jeans")), dhash(make_image("stripes"))) > 12


def test_check_images_flags_the_odd_one_out():
    res = check_images({
        "A": make_image("jeans"),
        "B": make_image("jeans", size=(800, 800), quality=50),
        "C": make_image("jeans", size=(300, 300), fmt="PNG"),
        "D": make_image("tee"),
    })
    assert [k for k, v in res.items() if v["match"] is False] == ["D"]
    assert all(res[k]["match"] is True for k in "ABC")


def test_check_images_needs_three_and_survives_garbage():
    two = check_images({"A": make_image("jeans"), "B": make_image("tee")})
    assert all(v["match"] is None for v in two.values())
    bad = check_images({"A": make_image("jeans"), "B": b"not an image", "C": make_image("jeans"), "D": make_image("jeans")})
    assert bad["B"]["match"] is None and bad["A"]["match"] is True


def test_transparent_png_does_not_crash():
    im = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    buf = io.BytesIO(); im.save(buf, "PNG")
    assert isinstance(dhash(buf.getvalue()), int)


def test_parse_meta():
    m = parse_meta('<meta property="og:title" content="X"><meta name="Description" content="Y">')
    assert m == {"og:title": "X", "description": "Y"}


class FakeImageSource:
    """Source that returns offers whose image_url points at a local server."""
    name = "fake"
    warnings: list = []

    def __init__(self, base):
        self.base = base

    def search(self, product):
        spec = [("A", "jeans"), ("B", "jeans-big"), ("C", "jeans-png"), ("D", "tee")]
        return [Offer(store=s, title="Levi's Men's 511 Jeans", price=2500 + i * 100, image_url=f"{self.base}/img/{k}.img")
                for i, (s, k) in enumerate(spec)]


@pytest.fixture
def local_site(tmp_path, monkeypatch):
    """Serve tests/pages plus generated images on localhost."""
    root = tmp_path / "www"
    shutil.copytree("tests/pages", root)
    (root / "img").mkdir()
    (root / "img" / "jeans.img").write_bytes(make_image("jeans"))
    (root / "img" / "jeans-big.img").write_bytes(make_image("jeans", size=(800, 800), quality=50))
    (root / "img" / "jeans-png.img").write_bytes(make_image("jeans", size=(300, 300), fmt="PNG"))
    (root / "img" / "tee.img").write_bytes(make_image("tee"))
    for name in "abc":
        (root / "img" / f"{name}.png").write_bytes(make_image("jeans", fmt="PNG"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def jeans_config(**extra):
    return {"products": [{"id": "j", "name": "x", "category": "clothing", "gender": "men",
                          "must_include": ["levis|levi", "511"]}], **extra}


def test_runner_image_flag_mode(tmp_path, local_site, monkeypatch):
    import pricewatch.runner as runner
    monkeypatch.setattr(runner, "build", lambda name, **kw: FakeImageSource(local_site))
    cfg = jeans_config(image_check={"mode": "flag", "delay_seconds": 0})
    data = run(cfg, History(tmp_path / "h.jsonl"), source_overrides=["fake"])
    rows = {r["store"]: r for r in data["products"][0]["summary"]["stores"]}
    assert rows["D"]["image_match"] is False and rows["A"]["image_match"] is True
    assert len(rows) == 4  # flag mode keeps everything


def test_runner_image_reject_mode(tmp_path, local_site, monkeypatch):
    import pricewatch.runner as runner
    monkeypatch.setattr(runner, "build", lambda name, **kw: FakeImageSource(local_site))
    cfg = jeans_config(image_check={"mode": "reject", "delay_seconds": 0})
    data = run(cfg, History(tmp_path / "h.jsonl"), source_overrides=["fake"])
    p = data["products"][0]
    assert {r["store"] for r in p["summary"]["stores"]} == {"A", "B", "C"}
    assert any(r["store"] == "D" and "image differs" in r["reason"] for r in p["rejected"])


# ---------- browser source (needs Playwright + a Chrome; skipped when unavailable) ----------
def _chrome():
    return os.environ.get("PRICEWATCH_CHROME") or (shutil.which("google-chrome") or shutil.which("chromium") or None)


@pytest.fixture
def browser_source():
    pytest.importorskip("playwright")
    from pricewatch.sources.browser import BrowserSource
    return lambda **kw: BrowserSource(delay_seconds=0, timeout_ms=8000, settle_ms=200, executable_path=_chrome(), **kw)


def test_browser_selectors_jsonld_and_og_fallbacks(local_site, browser_source):
    stores = {"Sel": {"title": ".pdp-title", "price": ".price", "mrp": ".mrp", "image": ".hero"}}
    product = Product(id="j", name="x", pages={
        "Sel": f"{local_site}/selectors.html", "Ld": f"{local_site}/jsonld.html", "Og": f"{local_site}/og.html"})
    try:
        src = browser_source(stores=stores)
        offers = {o.store: o for o in src.search(product)}
    except Exception as exc:  # no usable browser on this machine
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
            pytest.skip("no Chromium available")
        raise
    sel, ld, og = offers["Sel"], offers["Ld"], offers["Og"]
    assert (sel.price, sel.mrp, sel.title) == (2799.0, 4499.0, "Levi's Men's 511 Slim Fit Jeans")
    assert sel.image_url.endswith("/img/a.png")
    assert (ld.price, ld.mrp, ld.in_stock) == (2399.0, None, True)  # JSON-LD fallback
    assert ld.image_url == f"{local_site}/img/b.png"  # relative URL resolved against the page
    assert (og.price, og.title) == (3199.0, "Levi's 511 Slim Jeans for Men")  # OpenGraph fallback


def test_browser_reports_pages_without_a_price(local_site, browser_source):
    product = Product(id="j", name="x", pages={"Empty": f"{local_site}/empty.html", "Missing": f"{local_site}/nope.html"})
    try:
        src = browser_source()
        with pytest.raises(RuntimeError) as err:
            src.search(product)
    except Exception as exc:
        if "Executable doesn't exist" in str(exc):
            pytest.skip("no Chromium available")
        raise
    assert "Empty: could not find a price" in str(err.value) and "Missing: HTTP 404" in str(err.value)
