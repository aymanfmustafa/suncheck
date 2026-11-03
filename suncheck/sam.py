"""SAM 3.1 through the Meta Model API: segment every match for a noun phrase in an image (or video) via POST /v1/responses.

The output is text, one line per frame: `<Nf>` followed by records `id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;x=..;y=..;data=H,W,~payload|>`.
Masks are base85 (not base64); here only boxes and scores are parsed unless `decode_masks` is asked for.
"""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import requests

from .describe import load_key

API_URL = "https://api.meta.ai/v1/responses"
MODEL = "sam-3.1"

BOX_RE = re.compile(
    r"(?P<id>\d+)<\|box;x1=(?P<x1>[-\d.]+);y1=(?P<y1>[-\d.]+);x2=(?P<x2>[-\d.]+);y2=(?P<y2>[-\d.]+)"
    r"(?:;w=(?P<w>[-\d.]+);h=(?P<h>[-\d.]+))?(?:;c=(?P<c>[-\d.]+))?\|>"
)
FRAME_RE = re.compile(r"<(\d+)f>")


@dataclass
class Detection:
    frame: int
    obj_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None


MAX_SIDE = 2048  # what the phone page sends; the mask resolution SAM returns is bounded anyway


def open_frame(path: str | Path, max_side: int = MAX_SIDE):
    """The photo as the pipeline sees it: EXIF-rotated, RGB, HEIC decoded, no side longer than `max_side`.

    Returns the Pillow image and the scale from the file's pixels to these. Everything downstream (SAM, the sky mask,
    the horizon, the page's frame copy) works in this one frame, so boxes and masks map back by a single factor.
    """
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # a frame cut short by a flaky upload is still a frame
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = 1.0
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    return im, scale


@lru_cache(maxsize=16)
def prepare_image(path: str, max_side: int = MAX_SIDE) -> tuple[bytes, float]:
    """JPEG bytes of `open_frame(path)` and its scale, cached: one photo is sent to SAM once per noun phrase.

    The raw file is never sent: SAM rejects HEIC, and a rotated JPEG would come back with boxes in the sensor's frame.
    """
    import io

    im, scale = open_frame(path, max_side)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), scale


def segment_image(path: str | Path, prompt: str, include_confidence: bool = True, mask_encoding: str = "one_bit",
                  timeout: int = 300) -> dict:
    """Raw API response for one image and one noun phrase. Boxes come back in the sent image's pixels: see `prepare_image`."""
    key = load_key()
    if not key:
        raise SystemExit("META_API_KEY is not set.")
    jpeg, _ = prepare_image(str(path))
    b64 = base64.b64encode(jpeg).decode()
    payload = {
        "model": MODEL,
        "stream": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }
        ],
        "metadata": {"mask_encoding": mask_encoding, "include_confidence": "true" if include_confidence else "false"},
    }
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise SystemExit(f"SAM {r.status_code}: {r.text[:800]}")
    return r.json()


def output_text(resp: dict) -> str:
    """Concatenate every text field in a Responses-API result."""
    parts: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("type") in ("output_text", "text") and isinstance(x.get("text"), str):
                parts.append(x["text"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    walk(resp.get("output", resp))
    return "\n".join(parts)


def parse_detections(text: str) -> list[Detection]:
    dets: list[Detection] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fm = FRAME_RE.match(line.strip())
        frame = int(fm.group(1)) if fm else 0
        for m in BOX_RE.finditer(line):
            dets.append(Detection(frame, int(m["id"]), float(m["x1"]), float(m["y1"]), float(m["x2"]), float(m["y2"]),
                                  float(m["c"]) if m["c"] else None))
    return dets


def segment_and_parse(path: str | Path, prompt: str, **kw) -> tuple[list[Detection], str, dict]:
    """Detections in the pixels of `open_frame(path)`, which is the frame every stage of the pipeline works in."""
    resp = segment_image(path, prompt, **kw)
    text = output_text(resp)
    return parse_detections(text), text, resp


MASK_RE = re.compile(r"(?P<id>\d+)<\|box;[^|]*\|><\|mask;x=(?P<x>\d+);y=(?P<y>\d+)(?:;c=[-\d.]+)?;data=(?P<h>\d+),(?P<w>\d+),(?P<payload>[^|]+)\|>")


def parse_masks(text: str) -> list[dict]:
    """Per object: id, mask height and width, and the raw payload (marker character included)."""
    out = []
    for line in text.splitlines():
        for m in MASK_RE.finditer(line):
            out.append({"id": int(m["id"]), "x": int(m["x"]), "y": int(m["y"]), "h": int(m["h"]), "w": int(m["w"]),
                        "payload": m["payload"]})
    return out


def decode_mask(payload: str, h: int, w: int):
    """Decode one mask payload to a (h, w) boolean array with Meta's Python parser (`uv add meta-sam-parser`).

    The wire format is Meta's own: a custom base85 framing over a range-coded bitmap, so the official parser is used
    rather than a re-implementation. Raises ImportError with instructions when the package is absent.
    """
    try:
        import meta_sam_parser as msp
    except ImportError as e:
        raise ImportError("meta-sam-parser is not installed: run `uv add meta-sam-parser` in suncheck/") from e
    import numpy as np

    encoding = "one_bit" if payload.startswith("!") else "lossless"
    mask = msp.SegmentationMask(encoding=encoding, payload=payload, width=w, height=h)
    raster = msp.decode_mask_to_raster(mask)  # bytes, row-major, one byte per pixel
    arr = np.frombuffer(bytes(raster), dtype=np.uint8)
    if arr.size != h * w:
        raise RuntimeError(f"decoded {arr.size} pixels for a {w}x{h} mask")
    return arr.reshape(h, w).astype(bool)


def full_mask(dets_and_masks, image_h: int, image_w: int):
    """Union of all decoded masks placed at their boxes in a full-size boolean image."""
    import numpy as np

    import cv2

    full = np.zeros((image_h, image_w), dtype=bool)
    for d, m in dets_and_masks:
        raster = decode_mask(m["payload"], m["h"], m["w"])
        # the mask covers the box but may come back at a lower resolution (observed 1024 wide for a 1792-wide box),
        # so it is resized to the box before placing
        x0, y0 = int(round(d.x1)), int(round(d.y1))
        bw, bh = int(round(d.x2 - d.x1)) + 1, int(round(d.y2 - d.y1)) + 1
        if (raster.shape[1], raster.shape[0]) != (bw, bh):
            raster = cv2.resize(raster.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_LINEAR) > 0
        hh = min(bh, image_h - y0)
        ww = min(bw, image_w - x0)
        if hh > 0 and ww > 0:
            full[y0:y0 + hh, x0:x0 + ww] |= raster[:hh, :ww]
    return full


def sky_mask_sam(path: str | Path, image_h: int, image_w: int, prompt: str = "sky"):
    dets, text, _ = segment_and_parse(path, prompt)
    masks = parse_masks(text)
    by_id = {m["id"]: m for m in masks}
    pairs = [(d, by_id[d.obj_id]) for d in dets if d.obj_id in by_id]
    return full_mask(pairs, image_h, image_w), dets


if __name__ == "__main__":
    import sys

    dets, text, resp = segment_and_parse(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in resp.items() if k != "output"}, indent=1)[:1500])
    print(f"{len(dets)} detections; text length {len(text)}")
    for d in dets[:10]:
        print(d)
