import numpy as np
import pytest
from PIL import Image

from pricewatch.finder import Request, find
from pricewatch.match import Product, explain
from pricewatch.models import Offer
from pricewatch.normalize import tokenize
from pricewatch.search import SearchError
from pricewatch.search.serpapi import SerpApiShopping, parse_results
from pricewatch.vision import CATEGORIES, COLORS, consensus_similarity, describe_photo

from tests.test_core import make_image  # reuses the image generator


# ---------- fakes ----------
class FakeProvider:
    name = "fake"

    def __init__(self, offers):
        self.offers, self.calls = offers, []

    def search(self, query):
        self.calls.append(query)
        return list(self.offers)


class FakeEmbedder:
    """Cheap deterministic 'embedding': a 6x6 greyscale thumbnail, normalised."""

    def embed_images(self, images):
        out = []
        for im in images:
            v = np.asarray(im.convert("L").resize((6, 6)), dtype=float).ravel()
            v = v - v.mean()
            v = np.pad(v, (0, 256 - v.size))
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(out)

    def embed_texts(self, texts):
        import zlib

        rng = lambda t: np.random.default_rng(zlib.crc32(t.encode()))
        v = np.array([rng(t).normal(size=256) for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def png_uri(kind):
    import base64

    return "data:image/png;base64," + base64.b64encode(make_image(kind, fmt="PNG")).decode()


def offer(store, title, price, mrp=None, img="jeans"):
    return Offer(store=store, title=title, price=price, mrp=mrp, image_url=png_uri(img))


JEANS = [
    offer("A", "Levi's Men's 511 Slim Fit Jeans", 2799, 4499),
    offer("B", "Levis 511 Slim Fit Jeans Men Blue", 2399, 3999),
    offer("C", "LEVI'S 511 Slim Jeans for Men", 3199, 3999),
    offer("D", "Levi's Women's 511 Slim Jeans", 2599),
    offer("E", "Levi's 511 Jeans Men Boys Kids", 1799),
    offer("F", "Levi's 501 Original Jeans Men", 3499),
    offer("G", "Levi's 511 Slim Jeans Men (refurbished)", 1500),
]


# ---------- tokenizing ----------
def test_apostrophes_do_not_split_brand_names():
    assert tokenize("Levi's Men's 511") == ["levis", "mens", "511"]
    assert tokenize("levis mens 511") == ["levis", "mens", "511"]


# ---------- soft matching for free-text queries ----------
def test_free_text_requires_model_numbers_but_not_every_word():
    p = Product(id="x", name="sony wh-1000xm5 black headphones", soft_share=0.6)
    assert explain(p, Offer("S", "Sony WH-1000XM5 Wireless Headphones", 25000)) is None  # no "black": ok
    assert explain(p, Offer("S", "Sony WH-1000XM4 Black Headphones", 20000)) is not None  # wrong model number
    # a title that never names the model number can't be verified, so it is rejected
    j = Product(id="j", name="levis 511 slim fit jeans", soft_share=0.5)
    assert explain(j, Offer("S", "Levi's Slim Men", 3000)) is not None
    assert explain(j, Offer("S", "Levi's Mens 513 Slim Straight Jeans", 3000)) is not None
    assert explain(j, Offer("S", "Men's 511 Slim Jeans", 3000)) is None  # brand missing is tolerated


# ---------- serpapi parsing ----------
SERP = {"shopping_results": [
    {"title": "Levi's 511 Jeans", "source": "StoreA", "price": "₹2,799.00", "extracted_price": 2799.0,
     "old_price": "₹4,499.00", "extracted_old_price": 4499.0, "product_link": "https://g/a", "thumbnail": "http://t/a"},
    {"title": "Levi's 511 Jeans (used)", "source": "StoreB", "price": "₹999", "second_hand_condition": "used"},
    {"title": "No price", "source": "StoreC"},
    {"title": "Bad old price", "source": "StoreD", "price": "₹500", "old_price": "₹400"},
], "categorized_shopping_results": [{"shopping_results": [
    {"title": "Levi's 511 Jeans X", "source": "StoreE", "extracted_price": 3000}]}]}


def test_parse_serpapi_results():
    offers = {o.store: o for o in parse_results(SERP)}
    assert set(offers) == {"StoreA", "StoreB", "StoreD", "StoreE"}  # StoreC has no price
    a = offers["StoreA"]
    assert (a.price, a.mrp, a.url, a.image_url) == (2799.0, 4499.0, "https://g/a", "http://t/a")
    assert "(used)" in offers["StoreB"].title
    assert offers["StoreD"].mrp is None  # old price below price is not an MRP


def test_serpapi_without_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    with pytest.raises(SearchError, match="SERPAPI_KEY"):
        SerpApiShopping().search("anything")


# ---------- pipeline ----------
def test_name_only_search_filters_and_computes_stats():
    out = find(Request(query="levis 511 slim fit jeans", category="clothing", gender="men"),
               FakeProvider(JEANS), FakeEmbedder())
    s = out["summary"]
    assert {r["store"] for r in s["stores"]} == {"A", "B", "C"}
    assert s["lowest"] == {"store": "B", "price": 2399, "url": ""}
    reasons = {r["store"]: r["reason"] for r in out["rejected"]}
    assert set(reasons) == {"D", "E", "F", "G"}  # women's, kids, 501, refurbished
    assert out["listings_found"] == 7


def test_consensus_flags_the_listing_with_a_different_picture(monkeypatch):
    monkeypatch.setattr("pricewatch.finder.CONSENSUS_FLAG_BELOW", 0.93)  # the fake embedder is coarser than CLIP
    offers = [offer("A", "Levis 511 Jeans", 2500), offer("B", "Levis 511 Jeans", 2600),
              offer("C", "Levis 511 Jeans", 2700), offer("D", "Levis 511 Jeans", 2800, img="tee")]
    out = find(Request(query="levis 511 jeans"), FakeProvider(offers), FakeEmbedder())
    flags = {r["store"]: r["looks_same"] for r in out["summary"]["stores"]}
    assert flags == {"A": True, "B": True, "C": True, "D": False}
    assert len(out["summary"]["stores"]) == 4  # name-only search flags, it does not remove


def test_photo_search_drops_listings_that_look_different():
    offers = [offer("A", "Levis 511 Jeans", 2500), offer("B", "Levis 511 Jeans", 2600),
              offer("D", "Levis 511 Jeans", 2000, img="tee")]
    photo = Image.open(__import__("io").BytesIO(make_image("jeans", fmt="PNG")))
    out = find(Request(query="levis 511 jeans", photo=photo, min_similarity=0.8),
               FakeProvider(offers), FakeEmbedder())
    assert {r["store"] for r in out["summary"]["stores"]} == {"A", "B"}
    assert any(r["store"] == "D" and "looks different" in r["reason"] for r in out["rejected"])
    assert out["summary"]["lowest"]["store"] == "A"  # the cheap listing with the wrong picture is not "lowest"


def test_photo_only_builds_query_from_the_picture():
    photo = Image.open(__import__("io").BytesIO(make_image("jeans", fmt="PNG")))
    provider = FakeProvider([])
    out = find(Request(photo=photo, gender="men"), provider, FakeEmbedder())
    assert out["photo_info"]["category"] in CATEGORIES and out["photo_info"]["color"] in COLORS
    assert provider.calls[0].startswith("men ") and out["summary"] is None


def test_describe_photo_picks_the_closest_labels():
    class Oracle:  # image embedding == the text embedding of the right label
        def __init__(self):
            self.base = FakeEmbedder()

        def embed_texts(self, texts):
            return self.base.embed_texts(texts)

        def embed_images(self, images):
            # a vector close to both the category phrase and the colour+category phrase
            both = self.embed_texts(["a photo of a sneakers", "a photo of a blue sneakers"]).sum(axis=0)
            return (both / np.linalg.norm(both))[None, :]
    info = describe_photo(Image.new("RGB", (8, 8)), Oracle(), gender="women")
    assert (info["category"], info["color"], info["query"]) == ("sneakers", "blue", "women blue sneakers")


def test_errors_and_missing_input():
    with pytest.raises(ValueError):
        find(Request(), FakeProvider([]), FakeEmbedder())

    class Boom(FakeEmbedder):
        def embed_images(self, images):
            raise RuntimeError("gpu on fire")
    out = find(Request(query="levis 511 jeans"), FakeProvider(JEANS[:3]), Boom())
    assert out["summary"]["store_count"] == 3 and any("failed" in n for n in out["notes"])
    out = find(Request(query="levis 511 jeans", photo=Image.new("RGB", (8, 8))), FakeProvider(JEANS[:3]), None)
    assert any("not available" in n for n in out["notes"]) and out["summary"]["store_count"] == 3


def test_consensus_similarity_is_one_for_identical_vectors():
    v = np.tile(np.array([[1.0, 0, 0]]), (4, 1))
    assert np.allclose(consensus_similarity(v), 1.0)


# ---------- SearchApi.io ----------
def test_parse_searchapi_results():
    from pricewatch.search.searchapi import parse_results as parse
    offers = {o.store: o for o in parse({"shopping_results": [
        {"title": "Levi's 511 Jeans", "seller": "StoreA", "extracted_price": 2799.0, "product_link": "https://g/a", "thumbnail": "data:x"},
        {"title": "Levi's 511 Jeans", "seller": "StoreB", "price": "₹999", "durability": "Pre-owned"},
        {"title": "no price", "seller": "StoreC"}]})}
    assert set(offers) == {"StoreA", "StoreB"} and offers["StoreA"].price == 2799.0
    assert "(Pre-owned)" in offers["StoreB"].title


def test_first_accepted_falls_through_on_wrong_service_key():
    from pricewatch.search import FirstAccepted, SearchAuthError

    class Bad:
        name = "bad"
        def search(self, q): raise SearchAuthError("no")
    good = FakeProvider([Offer("S", "x", 1)])
    chain = FirstAccepted([Bad(), good])
    assert len(chain.search("q")) == 1
    with pytest.raises(SearchError, match="both"):
        FirstAccepted([Bad(), Bad()]).search("q")


def test_result_carries_check_time_and_row_details():
    out = find(Request(query="levis 511 slim fit jeans", category="clothing", gender="men"), FakeProvider(JEANS), FakeEmbedder())
    assert out["checked_at"].endswith("+00:00")
    row = out["summary"]["stores"][0]
    assert row["checked_at"] == out["checked_at"] and row["title"] and row["image_url"].startswith("data:")
    stamped = find(Request(query="levis 511 slim fit jeans"), FakeProvider([]), None,
                   offers_cache=lambda q: (list(JEANS), "2026-11-18T10:00:00+00:00"))
    assert stamped["checked_at"] == "2026-11-18T10:00:00+00:00"
