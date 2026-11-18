from __future__ import annotations

from .base import Source
from .browser import BrowserSource
from .jsonld import JsonLdSource
from .sample import SampleSource
from .shopify import ShopifySource

REGISTRY: dict[str, type[Source]] = {
    "sample": SampleSource,
    "jsonld": JsonLdSource,
    "browser": BrowserSource,
    "shopify": ShopifySource,
}


def build(name: str, **kwargs) -> Source:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown source '{name}'. available: {sorted(REGISTRY)}") from None
