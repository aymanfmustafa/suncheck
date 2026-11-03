"""Hours of direct sun per month for a spot, given its horizon profile, and the drawing of the sun's paths over the photo."""

from __future__ import annotations

from datetime import date

import numpy as np

from .camera import CameraPose
from .horizon import HorizonProfile
from .solar import day_path, month_days

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FULL_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def direct_sun_minutes(lat: float, lon: float, day: date, tz_offset_hours: float, horizon: HorizonProfile,
                       step_minutes: int = 1, outside: str = "blocked"):
    """Minutes of direct sun on `day`, the first and last direct-sun times, and the minute-by-minute mask.

    `outside` says what to assume for directions the photos did not cover: "blocked" counts only sun inside the
    photographed arc (the honest number for a balcony or window), "open" assumes a flat horizon there (a garden)."""
    t, az, el = day_path(lat, lon, day, tz_offset_hours, step_minutes)
    lit = (el > 0.0) & (el > horizon.elevation_at(az, unknown=90.0 if outside == "blocked" else 0.0))
    minutes = int(lit.sum()) * step_minutes
    first = t[np.argmax(lit)] if lit.any() else None
    last = t[len(t) - 1 - np.argmax(lit[::-1])] if lit.any() else None
    up = el > 0
    daylight = int(up.sum()) * step_minutes
    sunrise = t[np.argmax(up)] if up.any() else None
    sunset = t[len(t) - 1 - np.argmax(up[::-1])] if up.any() else None
    return minutes, first, last, daylight, (t, az, el, lit), sunrise, sunset


def _hhmm(t) -> str | None:
    return str(t)[11:16] if t is not None else None


def monthly_table(lat: float, lon: float, year: int, tz_offset_hours: float, horizon: HorizonProfile,
                  outside: str = "blocked") -> list[dict]:
    rows = []
    for d in month_days(year):
        minutes, first, last, daylight, (t, az, el, lit), sunrise, sunset = direct_sun_minutes(lat, lon, d, tz_offset_hours, horizon, outside=outside)
        # the lit stretches of the day as [start, end] in minutes after local midnight (a tower can split the afternoon)
        intervals = []
        start = None
        mins = (t - t.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
        for k in range(len(lit)):
            if lit[k] and start is None:
                start = int(mins[k])
            if (not lit[k] or k == len(lit) - 1) and start is not None:
                intervals.append([start, int(mins[k])])
                start = None
        rows.append({
            "month": MONTHS[d.month - 1],
            "date": d.isoformat(),
            "direct_sun_hours": round(minutes / 60.0, 2),
            "daylight_hours": round(daylight / 60.0, 2),
            "first_sun": _hhmm(first),
            "last_sun": _hhmm(last),
            "sunrise": _hhmm(sunrise),
            "sunset": _hhmm(sunset),
            "sunrise_min": int(mins[np.argmax(el > 0)]) if (el > 0).any() else None,
            "sunset_min": int(mins[len(el) - 1 - np.argmax((el > 0)[::-1])]) if (el > 0).any() else None,
            "lit_intervals_min": intervals,
        })
    return rows


# path colours (RGB): low winter sun in blue, the equinoxes in teal, high summer sun in amber; today in white
_BLUE, _TEAL, _AMBER = (96, 160, 255), (52, 199, 170), (246, 170, 52)


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def month_colour(month: int) -> tuple[int, int, int]:
    """RGB for a month's path: December blue, March and September teal, June amber, the rest in between."""
    dist = abs(month - 6) / 6.0  # 0 in June, 1 in December
    return _lerp(_AMBER, _TEAL, dist * 2) if dist <= 0.5 else _lerp(_TEAL, _BLUE, (dist - 0.5) * 2)


def _path_points(pose: CameraPose, lat, lon, day, tz, horizon):
    t, az, el = day_path(lat, lon, day, tz, 2)
    u, v, vis = pose.azel_to_pixel(az, el)
    lit = (el > 0) & (el > (horizon.elevation_at(az) if horizon is not None else 0.0))
    ok = vis & (el > -1)
    return np.stack([u, v], axis=1), ok, lit


def _polyline(out, pts, ok, lit, colour_bgr, thick, dashed_when_blocked=True):
    import cv2

    for k in range(len(pts) - 1):
        if not (ok[k] and ok[k + 1]):
            continue
        p0 = tuple(np.round(pts[k]).astype(int))
        p1 = tuple(np.round(pts[k + 1]).astype(int))
        if lit[k] or not dashed_when_blocked:
            cv2.line(out, p0, p1, colour_bgr, thick, cv2.LINE_AA)
        elif k % 4 == 0:
            cv2.line(out, p0, p1, colour_bgr, max(1, thick // 2), cv2.LINE_AA)


def _font(size: int):
    from PIL import ImageFont

    try:
        from matplotlib import font_manager as fm

        return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def draw_overlay(image_bgr: np.ndarray, pose: CameraPose, lat: float, lon: float, year: int, tz_offset_hours: float,
                 horizon: HorizonProfile | None = None, sun_now=None, label_hours: list[dict] | None = None,
                 months: str = "key", today: date | None = None) -> np.ndarray:
    """The sun's paths drawn over the photo.

    months="key": December, March and June on the 21st, plus today, each labelled with its hours; months="all": the twelve
    21sts, labelled on December, March and June. Lit stretches are solid, stretches behind the skyline dashed. Lines sit
    on a dark halo so they read on sky and on glare alike; labels are pills set on the line.
    """
    import cv2
    from PIL import Image, ImageDraw

    out = image_bgr.copy()
    h, w = out.shape[:2]
    thick = max(3, int(round(min(h, w) / 260)))
    days = month_days(year)
    if months == "key":
        chosen = [(days[11], 12, "Dec"), (days[2], 3, "Mar"), (days[5], 6, "Jun")]
        if today is not None:
            chosen.append((today, 0, "Today"))
    else:
        chosen = [(d, d.month, MONTHS[d.month - 1]) for d in days]
    paths = []
    for d, m, name in chosen:
        pts, ok, lit = _path_points(pose, lat, lon, d, tz_offset_hours, horizon)
        rgb = (255, 255, 255) if m == 0 else month_colour(m)
        paths.append((pts, ok, lit, rgb, m, name))

    # halo: every line first in black on a copy, blended under the colour
    halo = out.copy()
    for pts, ok, lit, rgb, m, name in paths:
        _polyline(halo, pts, ok, lit, (0, 0, 0), int(thick * 2.4))
    if horizon is not None:
        known = ~np.isnan(horizon.el)
        hu, hv, hvis = pose.azel_to_pixel(horizon.az, np.where(known, horizon.el, 0.0))
        hpts = np.stack([hu, hv], axis=1)
        hok = hvis & known
        _polyline(halo, hpts, hok, np.ones(len(hpts), bool), (0, 0, 0), int(thick * 1.6), False)
    out = cv2.addWeighted(halo, 0.45, out, 0.55, 0)
    for pts, ok, lit, rgb, m, name in paths:
        _polyline(out, pts, ok, lit, rgb[::-1], thick)
    if horizon is not None:
        _polyline(out, hpts, hok, np.ones(len(hpts), bool), (255, 255, 255), max(1, int(thick * 0.6)), False)

    # labels as pills on the line, on staggered rows so neighbouring paths do not collide
    im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = max(14, int(round(min(h, w) / 28)))  # readable once the photo is shown at phone width
    font = _font(size)
    pad = int(size * 0.55)
    labelled = {12: 0, 3: 1, 6: 2, 0: 3}
    for pts, ok, lit, rgb, m, name in paths:
        if m not in labelled or not ok.any():
            continue
        row = 0.09 + 0.065 * labelled[m]
        j = int(np.argmin(np.where(ok, np.abs(pts[:, 1] - h * row), 1e9)))
        txt = name
        if label_hours and m:
            txt += f"  {label_hours[m - 1]['direct_sun_hours']:.1f} h"
        tw = draw.textlength(txt, font=font)
        bw, bh = int(tw + pad * 2 + size * 0.9), int(size + pad * 1.2)
        x = int(np.clip(pts[j, 0] - bw / 2, 6, w - bw - 6))
        y = int(np.clip(pts[j, 1] - bh / 2, 6, h - bh - 6))
        draw.rounded_rectangle([x, y, x + bw, y + bh], radius=bh // 2, fill=(22, 22, 20, 220))
        cx_dot, cy_dot = x + pad + size * 0.3, y + bh / 2
        r = size * 0.27
        draw.ellipse([cx_dot - r, cy_dot - r, cx_dot + r, cy_dot + r], fill=rgb + (255,))
        draw.text((x + pad + size * 0.9, y + bh / 2), txt, font=font, fill=(255, 255, 255, 255), anchor="lm")
    if sun_now is not None:
        su, sv, svis = pose.azel_to_pixel(np.array([sun_now[0]]), np.array([sun_now[1]]))
        if svis[0]:
            rr = thick * 5
            draw.ellipse([su[0] - rr - 2, sv[0] - rr - 2, su[0] + rr + 2, sv[0] + rr + 2], outline=(0, 0, 0, 140), width=thick + 2)
            draw.ellipse([su[0] - rr, sv[0] - rr, su[0] + rr, sv[0] + rr], outline=(255, 255, 255, 255), width=thick)
            txt = "Sun now"
            tw = draw.textlength(txt, font=font)
            bw, bh = int(tw + pad * 2), int(size + pad * 1.2)
            x = int(np.clip(su[0] + rr + size * 0.5, 6, w - bw - 6))
            y = int(np.clip(sv[0] - bh / 2, 6, h - bh - 6))
            draw.rounded_rectangle([x, y, x + bw, y + bh], radius=bh // 2, fill=(22, 22, 20, 220))
            draw.text((x + pad, y + bh / 2), txt, font=font, fill=(255, 255, 255, 255), anchor="lm")
    im = Image.alpha_composite(im, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
