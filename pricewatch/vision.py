"""Image understanding with a free Hugging Face model (CLIP).

Two jobs:
  1. photo -> search phrase ("blue jeans") by zero-shot classification against a word list,
  2. image <-> image similarity, to check that store listings show the same product.

CLIP maps images and text into one vector space, so "how alike are these two pictures" is just
the cosine of two vectors. The model is loaded lazily; it runs on CPU, so it works for anonymous visitors on a free Space.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"

CATEGORIES = [
    # Indian and Western womenswear
    "kurti", "short kurti", "long kurta", "kurta set", "anarkali", "saree", "lehenga", "salwar suit", "dupatta",
    "palazzo pants", "co-ord set", "tunic top", "camisole top", "crop top", "top", "blouse", "shirt", "t-shirt",
    "polo t-shirt", "dress", "maxi dress", "skirt", "jumpsuit", "leggings",
    # menswear and unisex
    "jeans", "trousers", "shorts", "track pants", "hoodie", "sweatshirt", "jacket", "sweater", "blazer",
    # shoes, bags, accessories, gadgets
    "sneakers", "running shoes", "sandals", "heels", "formal shoes", "boots", "handbag", "backpack", "wallet",
    "belt", "cap", "watch", "sunglasses", "headphones", "earbuds", "smartphone", "laptop", "water bottle",
]
COLORS = ["black", "white", "blue", "navy blue", "light blue", "teal", "red", "maroon", "pink", "peach", "orange",
          "yellow", "mustard", "green", "olive green", "grey", "beige", "brown", "purple", "cream"]
PATTERNS = ["solid plain", "floral print", "block print", "striped", "checked", "polka dot", "embroidered",
            "geometric print", "printed"]
TEMPLATES = ["a photo of a {}", "a photo of a person wearing a {}", "a product photo of a {}"]


class Embedder(Protocol):
    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


class ClipEmbedder:
    """openai/clip-vit-base-patch32 through transformers. Heavy imports happen on first use."""

    def __init__(self, model_id: str = CLIP_MODEL):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> "ClipEmbedder":
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            # CPU on purpose: CLIP-B/32 takes well under a second per image on CPU, and public
            # visitors have no ZeroGPU quota (anonymous calls hit ZeroGPUQuotaExceeded).
            self._model = CLIPModel.from_pretrained(self.model_id).eval()
        return self

    def _to_array(self, feats) -> np.ndarray:
        torch = self._torch
        if not isinstance(feats, torch.Tensor):  # newer transformers can return an output object
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        return _normalize(feats.detach().float().cpu().numpy())

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        self.load()
        torch = self._torch
        inputs = self._processor(images=[i.convert("RGB") for i in images], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self._to_array(self._model.get_image_features(**inputs))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.load()
        torch = self._torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self._to_array(self._model.get_text_features(**inputs))


# ---------- photo -> search phrase ----------
def _label_vecs(embedder: Embedder, labels: list[str], templates=TEMPLATES) -> np.ndarray:
    """One vector per label: the average over several phrasings, which is steadier than a single prompt."""
    vecs = [embedder.embed_texts([t.format(label) for label in labels]) for t in templates]
    return _normalize(np.mean(vecs, axis=0))


def describe_photo(image: Image.Image, embedder: Embedder, gender: str | None = None) -> dict:
    """Zero-shot guess of what the photo shows. Returns category, colour, pattern, a search query, alternatives.

    It names the *kind* of product, its colour and print. It cannot read a brand or model, so the
    result is a search phrase to edit, not an exact identification.
    """
    img = embedder.embed_images([image])[0]
    cat_scores = _label_vecs(embedder, CATEGORIES) @ img
    order = [int(i) for i in np.argsort(-cat_scores)]
    category = CATEGORIES[order[0]]
    alternatives = [CATEGORIES[i] for i in order[1:3]]
    color = COLORS[int(np.argmax(embedder.embed_texts([f"a photo of a {c} {category}" for c in COLORS]) @ img))]
    pattern = PATTERNS[int(np.argmax(embedder.embed_texts([f"a photo of a {p} {category}" for p in PATTERNS]) @ img))]
    parts = [color, "" if pattern in ("solid plain", "printed") else pattern, category]
    if gender in ("men", "women"):
        parts.insert(0, gender)
    return {"category": category, "color": color, "pattern": pattern, "alternatives": alternatives,
            "query": " ".join(p for p in parts if p)}


# ---------- image <-> image similarity ----------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def consensus_similarity(vectors: np.ndarray) -> np.ndarray:
    """Similarity of each image to the group's average look. Needs 3+ images to mean anything."""
    centroid = vectors.mean(axis=0)
    return np.array([cosine(v, centroid) for v in vectors])


def fetch_image(url: str, timeout: float = 10.0, max_bytes: int = 3_000_000) -> Image.Image | None:
    """Download a listing thumbnail (http(s) or data: URI). None on any problem."""
    try:
        if url.startswith("data:"):
            raw = base64.b64decode(url.split(",", 1)[1])
        elif url.startswith(("http://", "https://")):
            req = urllib.request.Request(url, headers={"User-Agent": "pricewatch/0.4"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                return None
        else:
            return None
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def fetch_images(urls: list[str]) -> list[Image.Image | None]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda u: fetch_image(u) if u else None, urls))
