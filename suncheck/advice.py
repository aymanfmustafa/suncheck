"""Muse Spark words the result: a verdict, what the spot suits, what it does not, one tip. The numbers are ours."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import requests

from .describe import API_URL, DEFAULT_MODEL, load_key


def _scene_jpeg_b64(path: Path, long_side: int = 1024) -> str:
    from PIL import Image, ImageOps

    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((long_side, long_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _blockers_text(blockers: dict | None) -> str:
    if not blockers:
        return ""
    from .blockers import describe

    return ("What blocks the sun, from SAM 3.1 segmentation of the photos by noun phrase, with the direct sun our geometry "
            "charged to each (hours per day lost on the 21st of the month):\n" + describe(blockers) + "\n")


def build_prompt(table: list[dict], coverage: dict, lat: float, lon: float, blockers: dict | None = None) -> str:
    rows = "\n".join(f"{r['month']}: {r['direct_sun_hours']:.1f} h of direct sun"
                     + (f", from {r['first_sun']} to {r['last_sun']}" if r["first_sun"] else "")
                     + f" (daylight {r['daylight_hours']:.1f} h)" for r in table)
    best = max(table, key=lambda r: r["direct_sun_hours"])
    worst = min(table, key=lambda r: r["direct_sun_hours"])
    if best["direct_sun_hours"] <= 0:
        extremes = "No direct sun in any month: say so plainly in the verdict instead of naming strongest and weakest months."
    else:
        extremes = f"Strongest month: {best['month']} ({best['direct_sun_hours']:.1f} h). Weakest: {worst['month']} ({worst['direct_sun_hours']:.1f} h)."
    outside = ("directions outside that arc are counted as blocked (a wall or the building itself)"
               if coverage.get("outside") == "blocked" else "directions outside that arc are assumed open")
    return (
        "You write the result card of a sun-check app. The user photographed the view from a spot (a balcony, a window "
        "or a garden corner). Our software traced the real skyline in the photos and computed, from the sun's path, the "
        "hours of direct sun the spot gets on the 21st of each month. Use only the numbers below; invent none.\n\n"
        f"Location: latitude {lat:.2f}, longitude {lon:.2f} (judge the climate from it).\n"
        f"Direct sun by month (24-hour clock, local time):\n{rows}\n"
        f"{extremes}\n"
        f"The photos cover {coverage.get('azimuth_deg', 0):.0f} degrees of the horizon, clockwise from compass "
        f"{coverage.get('from_az')} to {coverage.get('to_az')} (0 north, 90 east, 180 south, 270 west; a range such as "
        f"293 to 60 passes through north); {outside}.\n"
        f"{_blockers_text(blockers)}"
        "The attached photo is one of the user's frames: name in a few words what blocks the sun, using the blockers and "
        "hours listed where there are any (for example 'the tower to the west, 2 hours a day' or 'your own overhang in "
        "summer'); if nothing blocks it, say so.\n\n"
        "Write under 110 words of plain English, no markdown, exactly these five lines, each starting with its label:\n"
        "Verdict: one sentence on the kind of sun this spot gets (time of day, seasons), naming the strongest and weakest months with their hours.\n"
        "Blocked by: what blocks the sun, from the photo.\n"
        "Good for: three uses, comma-separated.\n"
        "Not for: three uses, comma-separated.\n"
        "Tip: one practical suggestion (plants, shade, laundry, furniture, solar, a pet's bed)."
    )


def _ask_spark(content: list[dict], model: str, timeout: int) -> str:
    key = load_key()
    if not key:
        raise RuntimeError("META_API_KEY is not set")
    payload = {"model": model, "max_tokens": 4000, "reasoning_effort": "low", "messages": [{"role": "user", "content": content}]}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    if r.status_code == 400:
        payload.pop("reasoning_effort")
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"].get("content") or ""
    if not text.strip():
        raise RuntimeError(f"empty answer, finish_reason={data['choices'][0].get('finish_reason')}")
    return text.strip()


def answer_question(report: dict, question: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    """A short answer to the user's question about their spot, from the measured numbers (never beyond them)."""
    table = report["monthly_direct_sun_hours"]
    rows = "\n".join(f"{r['month']}: {r['direct_sun_hours']:.1f} h" + (f" ({r['first_sun']} to {r['last_sun']})" if r["first_sun"] else "")
                     for r in table)
    place = report.get("place", {})
    cov = report.get("coverage", {})
    prompt = (
        "You answer a question about a spot (a balcony, window or garden corner) whose direct sun we measured from the "
        "user's own photos. Use only the data below and general knowledge about plants, heat and daylight; where the data "
        "cannot answer, say what would be needed. Under 70 words, plain English, no markdown, no bullet points.\n\n"
        f"Location: latitude {place.get('lat', 0):.2f}, longitude {place.get('lon', 0):.2f}.\n"
        f"Direct sun on the 21st of each month (local time):\n{rows}\n"
        f"Photos cover {cov.get('azimuth_deg', 0):.0f} degrees of the horizon, clockwise from compass {cov.get('from_az')} to {cov.get('to_az')}.\n"
        f"{_blockers_text(report.get('blockers'))}"
        f"Result card already shown to the user:\n{report.get('advice') or '(none)'}\n\n"
        f"Question: {question.strip()}"
    )
    return _ask_spark([{"type": "text", "text": prompt}], model, timeout)


def write_advice(table: list[dict], coverage: dict, lat: float, lon: float, scene_path: Path | None = None,
                 model: str = DEFAULT_MODEL, timeout: int = 180, blockers: dict | None = None) -> str:
    key = load_key()
    if not key:
        raise RuntimeError("META_API_KEY is not set")
    content: list[dict] = [{"type": "text", "text": build_prompt(table, coverage, lat, lon, blockers)}]
    if scene_path is not None:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_scene_jpeg_b64(Path(scene_path))}"}})
    payload = {"model": model, "max_tokens": 4000, "reasoning_effort": "low", "messages": [{"role": "user", "content": content}]}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    if r.status_code == 400:
        payload.pop("reasoning_effort")
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"].get("content") or ""
    if not text.strip():
        raise RuntimeError(f"empty answer, finish_reason={data['choices'][0].get('finish_reason')}")
    return text.strip()
