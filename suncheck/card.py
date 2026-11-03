"""The share card: the hero overlay cropped to 4:5 with the answer set under it, for the X post and the camera roll."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .hours import _font

W, H = 1080, 1350
BAND = 300


def make_card(overlay_bgr: np.ndarray, summary: dict, out_path: Path) -> None:
    import cv2
    from PIL import Image, ImageDraw

    h, w = overlay_bgr.shape[:2]
    # fill the width, keep the top of the frame (sky and skyline), crop the rest
    scale = W / w
    resized = cv2.resize(overlay_bgr, (W, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    if resized.shape[0] < H:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[: resized.shape[0]] = resized
        resized = canvas
    crop = resized[:H]
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
    band = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)
    for i in range(BAND + 120):
        a = int(235 * min(1.0, i / 120.0)) if i < 120 else 235
        draw.line([(0, H - BAND - 120 + i), (W, H - BAND - 120 + i)], fill=(8, 8, 9, a))
    im = Image.alpha_composite(im, band)
    draw = ImageDraw.Draw(im)
    head, sub = _font(60), _font(32)
    y = H - BAND + 36
    # wrap the headline to the width
    words, lines, cur = summary.get("headline", "").split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if draw.textlength(trial, font=head) > W - 120 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    for ln in lines[:2]:
        draw.text((60, y), ln, font=head, fill=(245, 242, 234, 255))
        y += 72
    sub_t = summary.get("subline", "")
    while draw.textlength(sub_t, font=sub) > W - 120 and len(sub_t) > 10:
        sub_t = sub_t[: sub_t.rfind(" ")]
    draw.text((60, y + 6), sub_t, font=sub, fill=(170, 167, 158, 255))
    draw.ellipse([60, H - 62, 78, H - 44], fill=(226, 169, 58, 255))
    draw.text((90, H - 66), "Suncheck", font=_font(30), fill=(245, 242, 234, 255))
    im.convert("RGB").save(out_path, "JPEG", quality=90)
