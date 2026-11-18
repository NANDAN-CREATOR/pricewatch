"""Image comparison without AI models: difference hashing (dHash) via Pillow.

Idea: shrink the picture to 9x8 greyscale and record whether each pixel is brighter than its
right neighbour -> a 64-bit fingerprint. The same photo re-sized, re-compressed or re-hosted
gives (almost) the same fingerprint; a different product gives a very different one.

Strengths: free, instant, no model. Works well when stores reuse the brand's own product photo.
Limits: a different photo of the same product (other angle / model) will look "different", and
plain white-background packshots of similar garments can look alike. So by default the result is
a *flag* on the page, not a hard rejection. Colour is ignored on purpose (colour variants of one
product are the same product).
"""

from __future__ import annotations

import io

from PIL import Image

HASH_SIZE = 8
BITS = HASH_SIZE * HASH_SIZE
DEFAULT_THRESHOLD = 10  # max differing bits (of 64) to call two images "the same"


def dhash(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as im:
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")  # transparent PNGs -> white background
        canvas.alpha_composite(rgba)
        grey = canvas.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
        px = grey.tobytes()
    bits = 0
    for row in range(HASH_SIZE):
        for col in range(HASH_SIZE):
            i = row * (HASH_SIZE + 1) + col
            bits = (bits << 1) | (1 if px[i] > px[i + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def medoid(hashes: dict[str, int]) -> str:
    """The image closest to all the others: our stand-in for 'what this product looks like'."""
    return min(hashes, key=lambda k: sum(hamming(hashes[k], h) for h in hashes.values()))


def check_images(images: dict[str, bytes], threshold: int = DEFAULT_THRESHOLD) -> dict[str, dict]:
    """images: key (store) -> raw image bytes.

    Returns key -> {"distance": int|None, "match": bool|None}.
    match is None when there is no verdict (unreadable image, or fewer than 3 images so no consensus).
    """
    hashes: dict[str, int] = {}
    for key, data in images.items():
        try:
            hashes[key] = dhash(data)
        except Exception:
            pass
    out: dict[str, dict] = {k: {"distance": None, "match": None} for k in images}
    if len(hashes) < 3:
        return out
    ref = hashes[medoid(hashes)]
    for key, h in hashes.items():
        d = hamming(h, ref)
        out[key] = {"distance": d, "match": d <= threshold}
    return out
