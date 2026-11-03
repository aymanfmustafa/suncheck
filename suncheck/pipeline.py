"""One spot, several photos: camera poses, the sun check, the horizon, the hours and the overlays.

Shared by the command line (photos with EXIF) and the capture page (frames with the phone's own sensor readings).
The phone's compass is never trusted on its own: where the sun is in a frame, the sun fixes that frame's heading (and
the lens), and the difference to the compass is applied to the frames without the sun.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .camera import CameraPose, f_px_for_frame, solve_yaw_f_from_sun, solve_yaw_pitch_from_sun
from .horizon import HorizonProfile, brightest_blob, profile_from_photo, sky_mask_fallback
from .hours import FULL_MONTHS, draw_overlay, monthly_table
from .solar import sun_position


def _minutes(hhmm: str | None) -> int | None:
    return None if not hhmm else int(hhmm[:2]) * 60 + int(hhmm[3:5])


def _words_time(m: int) -> str:
    hh, mm = divmod(int(round(m / 15.0) * 15), 60)
    return f"{hh:02d}:{mm:02d}"


def summarise(table: list[dict]) -> dict:
    """The answer in words, from the monthly table: a headline and one line under it."""
    hours = [r["direct_sun_hours"] for r in table]
    lo, hi = min(hours), max(hours)
    best = table[int(np.argmax(hours))]
    worst = table[int(np.argmin(hours))]
    if hi <= 0:
        return {"headline": "No direct sun", "subline": "The sky is blocked in every direction the photos cover.",
                "time_of_day": "none", "hours_min": 0, "hours_max": 0, "best_month": None, "worst_month": worst["month"]}
    lit = [r for r in table if r["first_sun"]]
    firsts = sorted(_minutes(r["first_sun"]) for r in lit)
    lasts = sorted(_minutes(r["last_sun"]) for r in lit)
    f, l = firsts[len(firsts) // 2], lasts[len(lasts) // 2]
    if f >= 12 * 60:
        when = "Afternoon sun"
    elif l <= 13 * 60:
        when = "Morning sun"
    elif l - f >= 7 * 60:
        when = "Sun most of the day"
    else:
        when = "Midday sun"
    lo_r, hi_r = int(round(lo)), int(round(hi))
    if hi_r == 0:
        amount = "under an hour a day"
    elif lo_r == hi_r:
        amount = f"about {hi_r} hour{'s' if hi_r != 1 else ''} a day"
    elif lo_r == 0:
        amount = f"up to {hi_r} hours a day"
    else:
        amount = f"{lo_r} to {hi_r} hours a day"
    from_sunrise = sum(1 for r in lit if abs(_minutes(r["first_sun"]) - _minutes(r["sunrise"])) <= 12) >= len(lit) * 0.7
    to_sunset = sum(1 for r in lit if abs(_minutes(r["last_sun"]) - _minutes(r["sunset"])) <= 12) >= len(lit) * 0.7
    start = "from sunrise" if from_sunrise else f"from about {_words_time(f)}"
    end = "until sunset" if to_sunset else f"until about {_words_time(l)}"
    bi, wi = table.index(best), table.index(worst)
    subline = f"{start[0].upper() + start[1:]} {end}. Most in {FULL_MONTHS[bi]}, least in {FULL_MONTHS[wi]}."
    return {"headline": f"{when}, {amount}", "subline": subline, "time_of_day": when, "hours_min": lo, "hours_max": hi,
            "best_month": best["month"], "worst_month": worst["month"]}

DEFAULT_F35 = 26.0  # phone main cameras are 24 to 28 mm equivalent; refined from the sun when it is in frame
SUN_MIN_SCORE = 0.5
SUN_MAX_DISAGREEMENT_DEG = 40.0  # a "sun" further than this from where the sensors put it is a lamp or a reflection
MIN_SUN_OFF_AXIS_DEG = 12.0  # nearer the frame centre than this, the sun says too little about the lens


@dataclass
class PhotoInput:
    path: Path
    name: str = ""
    lat: float | None = None
    lon: float | None = None
    tz: float | None = None  # hours east of UTC, for the clock times in the table
    taken_utc: datetime | None = None
    time_source: str = ""
    f35: float | None = None
    f_px: float | None = None
    yaw: float | None = None  # heading of the optical axis, degrees clockwise from north
    pitch: float | None = None
    roll: float | None = None
    tilt_trusted: bool = False  # pitch and roll come from an accelerometer: keep them, solve heading and lens from the sun
    heading_source: str = ""
    sun_pixel: tuple[float, float] | None = None
    sensors: dict = field(default_factory=dict)


@dataclass
class Options:
    outside: str = "blocked"
    blockers: bool = True  # name what blocks the sun (SAM 3.1 noun phrases) and charge the lost minutes to it
    use_sam: bool = True
    pose: str = "sun"  # "sun": where the sun is in frame its solved pose replaces the sensor pose; "sensors": never
    advice: bool = False
    workers: int = 6  # SAM calls in flight at once: sky and sun per frame plus one per blocker phrase
    year: int | None = None


def load_image(path: Path) -> np.ndarray:
    """The photo as every stage sees it (EXIF-rotated, at most 2048 px on the long side: see `sam.open_frame`)."""
    from .sam import open_frame

    im, _ = open_frame(path)
    return np.asarray(im)


def photo_from_file(path: Path, lat=None, lon=None, tz=None, taken_local: str | None = None, f35=None, yaw=None,
                    pitch=None, roll=None, declination: float = 0.0, sun_pixel=None) -> PhotoInput:
    """A PhotoInput from a photo file's EXIF, with explicit values taking precedence."""
    from .exifmeta import pitch_roll_from_acceleration, read_meta

    meta = read_meta(path)
    p = PhotoInput(path=path, name=path.name, lat=lat if lat is not None else meta.lat,
                   lon=lon if lon is not None else meta.lon, tz=tz if tz is not None else meta.tz_offset_hours,
                   f35=f35 if f35 is not None else meta.f35_mm, sun_pixel=sun_pixel,
                   sensors={"exif_heading": meta.heading_deg, "heading_ref": meta.heading_ref,
                            "acceleration": meta.acceleration, "orientation": meta.orientation, "camera": meta.make_model})
    if p.tz is None and p.lon is not None:
        p.tz = float(round(p.lon / 15.0))  # older phones write no zone offset: the longitude's zone is the honest guess
        p.sensors["tz_source"] = "longitude"
    if taken_local:
        tz_used = p.tz if p.tz is not None else 4.0
        p.taken_utc = (datetime.fromisoformat(taken_local) - timedelta(hours=tz_used)).replace(tzinfo=timezone.utc)
        p.time_source = "given"
    elif meta.taken_utc:
        p.taken_utc, p.time_source = meta.taken_utc, "exif"
    elif meta.taken_local and p.tz is not None:
        p.taken_utc = (meta.taken_local - timedelta(hours=p.tz)).replace(tzinfo=timezone.utc)
        p.time_source = "exif clock, zone from longitude"
    if yaw is not None:
        p.yaw, p.heading_source = yaw, "given"
    elif meta.heading_deg is not None:
        p.yaw = (meta.heading_deg + (declination if meta.heading_ref == "M" else 0.0)) % 360.0
        p.heading_source = "exif compass"
    if pitch is not None or roll is not None:
        p.pitch, p.roll = pitch or 0.0, roll or 0.0
    elif meta.acceleration:
        p.pitch, p.roll = pitch_roll_from_acceleration(meta.acceleration, meta.orientation)
        p.sensors["tilt_source"] = "exif acceleration (unverified)"
    return p


def angle_between(az1, el1, az2, el2) -> float:
    a1, e1, a2, e2 = (np.radians(float(x)) for x in (az1, el1, az2, el2))
    c = np.cos(e1) * np.cos(e2) * np.cos(a1 - a2) + np.sin(e1) * np.sin(e2)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _circular_mean(deg: list[float]) -> float:
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


FRAME_MAX = 1600  # long side of the copies the page works with


def _frame_assets(fr: dict, out_dir: Path, stem: str) -> None:
    """What the page needs to animate a frame: a JPEG copy, the sky mask as PNG, the skyline as a polyline (copy pixels)."""
    import cv2

    from .horizon import skyline_from_mask

    h, w = fr["h"], fr["w"]
    scale = min(1.0, FRAME_MAX / max(h, w))
    cw, ch = int(round(w * scale)), int(round(h * scale))
    bgr = cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    cv2.imwrite(str(out_dir / f"{stem}_frame.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 86])
    sky = cv2.resize(fr["sky"].astype(np.uint8) * 255, (cw, ch), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / f"{stem}_sky.png"), sky)
    rows = skyline_from_mask(sky > 0)
    pts = [[int(u), int(rows[u])] for u in range(0, cw, max(1, cw // 240)) if rows[u] >= 0]
    fr["scale"] = scale
    fr["entry"].update({"frame": f"{stem}_frame.jpg", "frame_size": [cw, ch], "sky_png": f"{stem}_sky.png", "skyline_px": pts})


def _sun_paths(fr: dict, lat: float, lon: float, year: int, tz: float, horizon: HorizonProfile, outside: str, today) -> dict:
    """The sun's pixel position through the day, every five minutes, for the 21st of each month and for today: for the
    living photo. Each point is [u, v, in_frame, lit] in the frame copy's pixels."""
    from .hours import direct_sun_minutes
    from .solar import month_days

    pose, scale = fr["final"], fr["scale"]
    out = {"step_min": 5, "months": {}, "up": {}}
    days = [(d.month, d) for d in month_days(year)] + [(0, today)]
    for key, day in days:
        _, _, _, _, (t, az, el, lit), _, _ = direct_sun_minutes(lat, lon, day, tz, horizon, step_minutes=5, outside=outside)
        u, v, vis = pose.azel_to_pixel(az, el)
        pts = [[round(float(u[i]) * scale, 1), round(float(v[i]) * scale, 1), int(bool(vis[i]) and el[i] > -0.8), int(bool(lit[i]))]
               for i in range(len(t))]
        out["months"][str(key)] = pts
        up = np.flatnonzero(el > 0)
        out["up"][str(key)] = [int(up[0]) * 5, int(up[-1]) * 5] if len(up) else None
    return out


def _sam_sun(path: Path):
    from .sam import segment_and_parse

    dets, _, _ = segment_and_parse(path, "the sun")
    dets = [d for d in dets if (d.score or 1.0) >= SUN_MIN_SCORE]
    if not dets:
        return None, "sam-3.1 (no sun)"
    best = max(dets, key=lambda d: d.score or 0)
    return ((best.x1 + best.x2) / 2, (best.y1 + best.y2) / 2), f"sam-3.1 ({best.score:.2f})"


def _sam_sky(path: Path, h: int, w: int):
    from .sam import sky_mask_sam

    mask, dets = sky_mask_sam(path, h, w)
    return mask, f"sam-3.1 ({len(dets)} regions)"


def analyze_set(photos: list[PhotoInput], opts: Options, out_dir: Path, progress=None) -> dict:
    """Run the whole chain on one spot's photos; writes overlays and report.json into out_dir and returns the report."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"photos": [], "notes": [], "calibration": {}}

    def say(text: str, frac: float, **extra):
        if progress:
            progress(text, frac, **extra)

    # place and time: the first photo that knows them speaks for the spot
    lat = next((p.lat for p in photos if p.lat is not None), None)
    lon = next((p.lon for p in photos if p.lon is not None), None)
    tz = next((p.tz for p in photos if p.tz is not None), None)
    if lat is None or lon is None:
        raise ValueError("no location: the photos carry no GPS and none was given")
    if tz is None:
        tz = float(round(lon / 15.0))
        report["notes"].append(f"no time-zone offset in the photos; clock times use the zone of the longitude (UTC{tz:+.0f})")
    report["place"] = {"lat": lat, "lon": lon, "tz": tz}

    # load frames, settle the lens and the time
    frames: list[dict] = []
    from .sam import open_frame

    for p in photos:
        im, load_scale = open_frame(p.path)
        rgb = np.asarray(im)
        h, w = rgb.shape[:2]
        if p.sun_pixel is not None and load_scale != 1.0:
            p.sun_pixel = (p.sun_pixel[0] * load_scale, p.sun_pixel[1] * load_scale)
        if p.f_px:
            f_px, f_source = p.f_px, "given"
        elif p.f35:
            f_px, f_source = f_px_for_frame(p.f35, w, h), "exif" if p.sensors.get("camera") else "device"
        else:
            f_px, f_source = f_px_for_frame(DEFAULT_F35, w, h), "default"
        if p.taken_utc is None:
            taken = datetime.now(timezone.utc)
            report["notes"].append(f"{p.name}: no capture time; the sun is placed for now")
            time_source = "now"
        else:
            taken, time_source = p.taken_utc, p.time_source or "given"
        az_s, el_s = sun_position(lat, lon, np.datetime64(taken.replace(tzinfo=None)))
        frames.append({"p": p, "rgb": rgb, "w": w, "h": h, "f_px": float(f_px), "f_source": f_source, "taken": taken,
                       "time_source": time_source, "sun_az": float(az_s), "sun_el": float(el_s)})

    # SAM 3.1: the sun and the sky, every frame at once
    from .blockers import BLOCKER_PHRASES, attribute, label_map, segment_blockers

    say("Finding the sky, the sun and what blocks it" if opts.blockers else "Finding the sky and the sun in each photo", 0.05)
    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, opts.workers)) as ex:
        for i, fr in enumerate(frames):
            p = fr["p"]
            if p.sun_pixel is not None:
                fr["sun_uv"], fr["sun_source"] = tuple(float(x) for x in p.sun_pixel), "given"
            elif opts.use_sam and fr["sun_el"] > 0:
                tasks.append((i, "sun", ex.submit(_sam_sun, p.path)))
            else:
                fr["sun_uv"], fr["sun_source"] = None, "not looked for" if fr["sun_el"] <= 0 else "sam off"
            if opts.use_sam:
                tasks.append((i, "sky", ex.submit(_sam_sky, p.path, fr["h"], fr["w"])))
            if opts.use_sam and opts.blockers:
                for phrase in BLOCKER_PHRASES:
                    tasks.append((i, f"blocker:{phrase}", ex.submit(segment_blockers, p.path, fr["h"], fr["w"], phrase)))
        done = 0
        for i, kind, fut in tasks:
            fr = frames[i]
            try:
                res = fut.result()
                if kind == "sun":
                    fr["sun_uv"], fr["sun_source"] = res
                elif kind == "sky":
                    fr["sky"], fr["sky_source"] = res
                else:
                    fr.setdefault("instances", []).extend(res)
            except Exception as e:  # noqa: BLE001
                report["notes"].append(f"{fr['p'].name}: SAM {kind} failed ({type(e).__name__}: {str(e)[:100]})")
                if kind == "sun":
                    gray = cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2GRAY)
                    blob = brightest_blob(gray)
                    fr["sun_uv"], fr["sun_source"] = blob, "brightest blob" if blob else "none"
            done += 1
            what = "the sky, the sun and what blocks it" if opts.blockers else "the sky and the sun"
            say(f"Finding {what} ({done} of {len(tasks)})", 0.05 + 0.6 * done / max(1, len(tasks)))
    for fr in frames:
        if "sky" not in fr:
            fr["sky"], fr["sky_source"] = sky_mask_fallback(fr["rgb"]), "colour fallback"
        fr.setdefault("sun_uv", None)
        fr.setdefault("sun_source", "none")
        fr.setdefault("instances", [])
        fr["labels"] = label_map(fr["instances"], fr["sky"])
    # the page can start tracing skylines while the rest is computed
    partial = []
    for fr in frames:
        fr["entry"] = {"file": fr["p"].name}
        _frame_assets(fr, out_dir, Path(fr["p"].name).stem or "photo")
        sun_px = [fr["sun_uv"][0] * fr["scale"], fr["sun_uv"][1] * fr["scale"]] if fr.get("sun_uv") else None
        partial.append({"file": fr["p"].name, "frame": fr["entry"]["frame"], "frame_size": fr["entry"]["frame_size"],
                        "skyline_px": fr["entry"]["skyline_px"], "sun_px": sun_px})
    say("Solving camera headings from the sun", 0.68, partial=partial)

    # poses: sensors first, then the sun where it is in frame
    say("Solving camera headings from the sun", 0.7)
    yaw_corrections: list[float] = []
    lens_ratios: list[float] = []
    for fr in frames:
        p, w, h = fr["p"], fr["w"], fr["h"]
        base_yaw = p.yaw if p.yaw is not None else 180.0
        base = CameraPose(fr["f_px"], w, h, base_yaw, p.pitch or 0.0, p.roll or 0.0)
        fr["base"] = base
        fr["entry"] |= {
            "file": p.name, "size": [w, h], "f_px": fr["f_px"], "f_source": fr["f_source"], "lat": p.lat, "lon": p.lon,
            "taken_utc": fr["taken"].isoformat(), "time_source": fr["time_source"], "heading_source": p.heading_source or "none",
            "pose_sensors": {"yaw": p.yaw, "pitch": p.pitch, "roll": p.roll} if p.yaw is not None or p.pitch is not None else None,
            "sun_now": {"az": fr["sun_az"], "el": fr["sun_el"]},
            "sun_detection": {"pixel": list(fr["sun_uv"]) if fr["sun_uv"] else None, "source": fr["sun_source"]},
            "sun_check": None, "pose_from_sun": None, "sensors": p.sensors,
        }
        fr["final"] = base
        fr["final_source"] = "sensors" if p.yaw is not None else "no heading (assumed south)"
        if fr["sun_uv"] is None or fr["sun_el"] <= 0:
            continue
        u, v = fr["sun_uv"]
        az_det, el_det = base.pixel_to_azel(u, v)
        err = angle_between(fr["sun_az"], fr["sun_el"], az_det, el_det)
        if p.yaw is not None:
            if err > SUN_MAX_DISAGREEMENT_DEG:
                report["notes"].append(f"{p.name}: the bright spot SAM called the sun is {err:.0f} deg from where the sensors put the sun; ignored")
                fr["entry"]["sun_detection"]["rejected"] = True
                continue
            u_pred, v_pred, vis = base.azel_to_pixel(np.array([fr["sun_az"]]), np.array([fr["sun_el"]]))
            fr["entry"]["sun_check"] = {"angular_error_deg": err, "detected_pixel": [u, v],
                                        "predicted_pixel": [float(u_pred[0]), float(v_pred[0])], "predicted_in_frame": bool(vis[0])}
            if p.sensors.get("compass") is not None and p.tilt_trusted:
                # the alternative reading of the phone's compass: its heading taken as the camera's heading directly
                alt = CameraPose(fr["f_px"], w, h, float(p.sensors["compass"]), p.pitch or 0.0, p.roll or 0.0)
                a2, e2 = alt.pixel_to_azel(u, v)
                fr["entry"]["sun_check"]["angular_error_deg_compass_direct"] = angle_between(fr["sun_az"], fr["sun_el"], a2, e2)
        off_axis = float(np.degrees(np.arctan(np.hypot(u - base.cx, v - base.cy) / fr["f_px"])))
        if p.tilt_trusted and p.pitch is not None and off_axis >= MIN_SUN_OFF_AXIS_DEG:
            solved = solve_yaw_f_from_sun(fr["f_px"], w, h, (u, v), fr["sun_az"], fr["sun_el"], p.pitch, p.roll or 0.0, yaw0=base_yaw)
            lens_solved = True
            if fr["f_source"] == "default":
                lens_ratios.append(solved.f_px / fr["f_px"])
        else:
            solved = solve_yaw_pitch_from_sun(fr["f_px"], w, h, (u, v), fr["sun_az"], fr["sun_el"], roll=p.roll or 0.0,
                                              yaw0=base_yaw, pitch0=p.pitch or 0.0)
            lens_solved = False
        fr["entry"]["pose_from_sun"] = {
            "yaw": solved.yaw, "pitch": solved.pitch, "roll": solved.roll, "f_px": solved.f_px, "lens_solved": lens_solved,
            "sun_off_axis_deg": off_axis, "fit_error_deg": getattr(solved, "sun_error_deg", None),
            "yaw_minus_sensors": _wrap(solved.yaw - p.yaw) if p.yaw is not None else None,
            "pitch_minus_sensors": solved.pitch - p.pitch if p.pitch is not None else None,
        }
        if p.yaw is not None:
            yaw_corrections.append(_wrap(solved.yaw - p.yaw))
        if opts.pose == "sun":
            fr["final"], fr["final_source"] = solved, "sun"

    # the session's calibration: what the sun taught about the compass and the lens, applied to sunless frames
    yaw_corr = _circular_mean(yaw_corrections) if yaw_corrections else None
    lens_ratio = float(np.median(lens_ratios)) if lens_ratios else None
    report["calibration"] = {"heading_correction_deg": yaw_corr, "lens_ratio": lens_ratio,
                             "from_frames": [fr["p"].name for fr in frames if fr["entry"]["pose_from_sun"]]}
    for fr in frames:
        if fr["final_source"] == "sun":
            continue
        p = fr["p"]
        f_px = fr["f_px"] * lens_ratio if (lens_ratio and fr["f_source"] == "default") else fr["f_px"]
        if p.yaw is not None and yaw_corr is not None:
            fr["final"] = CameraPose(f_px, fr["w"], fr["h"], (p.yaw + yaw_corr) % 360.0, p.pitch or 0.0, p.roll or 0.0)
            fr["final_source"] = "sensors + sun correction"
        elif f_px != fr["f_px"]:
            fr["final"] = CameraPose(f_px, fr["w"], fr["h"], fr["base"].yaw, fr["base"].pitch, fr["base"].roll)

    # horizons, hours, overlays
    say("Tracing the skyline and counting the hours", 0.8)
    merged = HorizonProfile.empty()
    year = opts.year or datetime.now().year
    for fr in frames:
        p, pose = fr["p"], fr["final"]
        prof = profile_from_photo(fr["sky"], pose)
        fr["prof"] = prof
        merged = merged.merge(prof)
        e = fr["entry"]
        e["pose_used"] = {"yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll, "f_px": pose.f_px, "source": fr["final_source"],
                          "hfov": pose.hfov_deg(), "vfov": pose.vfov_deg()}
        e["sky_source"], e["sky_fraction"] = fr["sky_source"], float(fr["sky"].mean())
        e["horizon_coverage_deg"] = prof.coverage()[0]
    # no sky in any frame means no evidence at all (indoors, or the camera pointed at the floor): never assume an open horizon
    no_sky = merged.coverage()[0] <= 0
    if no_sky:
        opts.outside = "blocked"
        report["notes"].append("no sky in any photo; the spot is counted as fully blocked")
    table = monthly_table(lat, lon, year, tz, merged, outside=opts.outside)
    today = frames[0]["taken"].astimezone(timezone(timedelta(hours=tz))).date()
    report["blockers"] = None
    if opts.blockers and opts.use_sam and not no_sky:
        say("Charging the lost sun to what blocks it", 0.84)
        try:
            report["blockers"] = attribute(frames, lat, lon, year, tz, merged, opts.outside)
        except Exception as e:  # noqa: BLE001
            report["notes"].append(f"blockers unavailable ({type(e).__name__}: {str(e)[:120]})")
    for fr in frames:
        p, pose = fr["p"], fr["final"]
        bgr = cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2BGR)
        sun_now = (fr["sun_az"], fr["sun_el"]) if fr["sun_el"] > 0 else None
        stem = Path(fr["p"].name).stem or "photo"
        over = draw_overlay(bgr, pose, lat, lon, year, tz, merged, sun_now=sun_now, label_hours=table, months="key", today=today)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 88])
        over_all = draw_overlay(bgr, pose, lat, lon, year, tz, merged, sun_now=sun_now, label_hours=table, months="all")
        cv2.imwrite(str(out_dir / f"{stem}_overlay_all.jpg"), over_all, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # the debug view: the sky mask tinted and the detected sun marked
        skyviz = bgr.copy()
        skyviz[fr["sky"]] = (0.5 * skyviz[fr["sky"]] + 0.5 * np.array([255, 200, 80])).astype(np.uint8)
        if fr["sun_uv"] is not None and not fr["entry"]["sun_detection"].get("rejected"):
            cv2.drawMarker(skyviz, (int(fr["sun_uv"][0]), int(fr["sun_uv"][1])), (255, 0, 255), cv2.MARKER_CROSS, 60, 4)
        cv2.imwrite(str(out_dir / f"{stem}_sky.jpg"), skyviz, [cv2.IMWRITE_JPEG_QUALITY, 80])
        fr["entry"]["overlay"], fr["entry"]["overlay_all"], fr["entry"]["sky_image"] = f"{stem}_overlay.jpg", f"{stem}_overlay_all.jpg", f"{stem}_sky.jpg"
        fr["entry"]["sun_paths"] = _sun_paths(fr, lat, lon, year, tz, merged, opts.outside, today)
        fr["entry"]["f_px_frame"] = pose.f_px * fr["scale"]  # so the page can draw the sun at its true angular size
        if fr["sun_uv"] is not None:
            fr["entry"]["sun_pixel_frame"] = [fr["sun_uv"][0] * fr["scale"], fr["sun_uv"][1] * fr["scale"]]
        fr["entry"]["blockers"] = fr.get("global_ids", [])
        report["photos"].append(fr["entry"])

    report["monthly_direct_sun_hours"] = table
    report["summary"] = summarise(table)
    if no_sky:
        report["summary"] |= {"headline": "No sky in these photos", "no_sky": True,
                              "subline": "Nothing in the frames is sky, so there is nothing to measure. Point the camera at the view, with the skyline in the shot."}
        opts.advice = False
    # the share card: the frame with the sun in it (else the first), the answer under it
    try:
        from .card import make_card

        hero = next((fr for fr in frames if fr["entry"].get("pose_from_sun")), frames[0])
        make_card(cv2.imread(str(out_dir / hero["entry"]["overlay"])), report["summary"], out_dir / "card.jpg")
        report["card"] = "card.jpg"
    except Exception as e:  # noqa: BLE001
        report["notes"].append(f"share card unavailable ({type(e).__name__}: {str(e)[:100]})")
    deg, az_first, az_last = merged.coverage()
    report["coverage"] = {"azimuth_deg": deg, "from_az": az_first, "to_az": az_last, "outside": opts.outside}
    report["horizon"] = {"az": merged.az.tolist(), "el": [None if np.isnan(x) else round(float(x), 2) for x in merged.el]}
    checks = [e["sun_check"]["angular_error_deg"] for e in report["photos"] if e["sun_check"]]
    report["sun_check_summary"] = {"frames_with_sun": sum(1 for e in report["photos"] if e["pose_from_sun"]),
                                   "sensor_errors_deg": checks, "worst_sensor_error_deg": max(checks) if checks else None}

    if opts.advice:
        say("Writing the verdict", 0.92)
        try:
            from .advice import write_advice

            scene = next((fr["p"].path for fr in frames if fr["sun_uv"] is not None), frames[0]["p"].path)
            report["advice"] = write_advice(table, report["coverage"], lat, lon, scene, blockers=report.get("blockers"))
        except Exception as e:  # noqa: BLE001
            report["advice"] = None
            report["notes"].append(f"advice unavailable ({type(e).__name__}: {str(e)[:120]})")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    say("Done", 1.0)
    return report
