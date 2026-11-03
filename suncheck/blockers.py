"""What blocks the sun, and what it costs: SAM 3.1 names each obstruction in the photos (building, tree, wall, railing,
overhang, hill), and our geometry charges every lost minute of every month to the one that hid the sun at that moment.

The sun's path for the 21st of each month is walked minute by minute. Where the horizon profile says the sun is blocked,
its direction is projected into the photo that covers it and the obstruction under that pixel takes the blame. Minutes
whose direction no photo covers, or that fall above a photo's top edge, are "outside the photos"; minutes on a pixel
that is not sky but matched no noun phrase are "other".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import CameraPose
from .horizon import HorizonProfile
from .solar import day_path, month_days

BLOCKER_PHRASES = ["building", "tree", "plant", "wall", "railing", "ceiling", "hill"]
KIND_NAMES = {"building": "Building", "tree": "Tree", "plant": "Plant", "wall": "Wall", "railing": "Railing", "ceiling": "Overhang",
              "hill": "Hill"}
MIN_MONTH_HOURS = 0.05  # an obstruction that never costs three minutes in any month is not reported
MIN_SCORE = 0.45  # SAM's confidence below which a match is ignored
MIN_AREA_FRACTION = 0.0005  # of the frame; a far tower on the skyline is tiny and still blocks the sunset
MERGE_OVERLAP = 0.5  # two matches of one kind in different photos are one object when their bearings overlap this much
MAX_OUTLINE_POINTS = 60
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def segment_blockers(path: str | Path, h: int, w: int, phrase: str) -> list[dict]:
    """Every match of one noun phrase in one photo: kind, score, box and a full-frame boolean mask."""
    from .sam import full_mask, parse_masks, segment_and_parse

    dets, text, _ = segment_and_parse(path, phrase)
    by_id = {m["id"]: m for m in parse_masks(text)}
    out = []
    for d in dets:
        if (d.score or 1.0) < MIN_SCORE or d.obj_id not in by_id:
            continue
        mask = full_mask([(d, by_id[d.obj_id])], h, w)
        if mask.mean() < MIN_AREA_FRACTION:
            continue
        out.append({"kind": phrase, "score": float(d.score or 1.0), "box": [d.x1, d.y1, d.x2, d.y2], "mask": mask})
    return out


def label_map(instances: list[dict], sky: np.ndarray) -> np.ndarray:
    """Per pixel, 1 + the index of the instance in front (smaller matches are painted over larger ones); 0 where none."""
    h, w = sky.shape
    labels = np.zeros((h, w), dtype=np.int32)
    order = sorted(range(len(instances)), key=lambda i: -float(instances[i]["mask"].sum()))
    for i in order:
        m = instances[i]["mask"] & ~sky
        labels[m] = i + 1
    return labels


def _bearing_range(inst: dict, pose: CameraPose) -> tuple[float, float]:
    """The compass arc an instance spans, read along the row through the middle of its box."""
    x1, y1, x2, y2 = inst["box"]
    yc = (y1 + y2) / 2.0
    az, _ = pose.pixel_to_azel(np.array([x1, x2]), np.array([yc, yc]))
    lo, hi = float(az[0]) % 360.0, float(az[1]) % 360.0
    if (hi - lo) % 360.0 > 180.0:  # the box's left edge is east of its right edge only when it crosses north
        lo, hi = hi, lo
    return lo, hi


def _arc_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Degrees shared by two clockwise arcs on the compass."""
    def unwrap(arc):
        lo, hi = arc
        return lo, hi if hi >= lo else hi + 360.0

    a0, a1 = unwrap(a)
    best = 0.0
    for shift in (-360.0, 0.0, 360.0):
        b0, b1 = unwrap(b)
        b0, b1 = b0 + shift, b1 + shift
        best = max(best, min(a1, b1) - max(a0, b0))
    return max(0.0, best)


def _outline(mask: np.ndarray, scale: float) -> list[list[int]]:
    """The largest contour of a mask, simplified, in the page's frame-copy pixels."""
    import cv2

    small = cv2.resize(mask.astype(np.uint8), (max(1, int(round(mask.shape[1] * scale))), max(1, int(round(mask.shape[0] * scale)))),
                       interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(poly) > MAX_OUTLINE_POINTS:
        poly = poly[np.linspace(0, len(poly) - 1, MAX_OUTLINE_POINTS).astype(int)]
    return [[int(x), int(y)] for x, y in poly]


def attribute(frames: list[dict], lat: float, lon: float, year: int, tz: float, horizon: HorizonProfile, outside: str) -> dict:
    """The blockers of a set of frames, each with the hours of direct sun it costs per day in every month.

    Each frame dict needs: "p" (with .name), "final" (CameraPose), "sky", "instances", "labels", "scale".
    """
    # one global blocker per physical object: instances of a kind whose bearings overlap are the same thing seen twice
    globals_: list[dict] = []
    for fi, fr in enumerate(frames):
        fr["global_ids"] = []
        for li, inst in enumerate(fr.get("instances", [])):
            arc = _bearing_range(inst, fr["final"])
            match = None
            for g in globals_:
                if g["kind"] != inst["kind"] or fi in g["frame_ids"]:
                    continue
                span = min(_arc_len(arc), _arc_len(g["arc"]))
                if span > 0 and _arc_overlap(arc, g["arc"]) >= MERGE_OVERLAP * span:
                    match = g
                    break
            if match is None:
                match = {"kind": inst["kind"], "arc": arc, "score": inst["score"], "frame_ids": set(), "frames": []}
                globals_.append(match)
            else:
                match["arc"] = _union_arc(match["arc"], arc)
                match["score"] = max(match["score"], inst["score"])
            match["frame_ids"].add(fi)
            match["frames"].append({"file": fr["p"].name, "outline": _outline(inst["mask"], fr["scale"]),
                                    "box": [round(v * fr["scale"], 1) for v in inst["box"]]})
            fr["global_ids"].append(globals_.index(match))

    n = len(globals_)
    minutes = np.zeros((12, n), dtype=int)
    other = np.zeros(12, dtype=int)
    outside_m = np.zeros(12, dtype=int)
    blocked_total = np.zeros(12, dtype=int)
    unknown_el = 90.0 if outside == "blocked" else 0.0
    for mi, day in enumerate(month_days(year)):
        _, az, el = day_path(lat, lon, day, tz, 1)
        up = el > 0.0
        lit = up & (el > horizon.elevation_at(az, unknown=unknown_el))
        blocked = up & ~lit
        blocked_total[mi] = int(blocked.sum())
        assigned = np.full(len(az), -2, dtype=int)  # -2 outside the photos, -1 other, >= 0 a blocker
        for fr in frames:
            u, v, vis = fr["final"].azel_to_pixel(az, el)
            inside = vis & blocked & (assigned == -2)
            if not inside.any():
                continue
            ui = np.clip(u[inside].astype(int), 0, fr["labels"].shape[1] - 1)
            vi = np.clip(v[inside].astype(int), 0, fr["labels"].shape[0] - 1)
            lab = fr["labels"][vi, ui]
            ids = np.array(fr["global_ids"], dtype=int)
            assigned[inside] = np.where(lab > 0, ids[np.clip(lab - 1, 0, max(0, len(ids) - 1))] if len(ids) else -1, -1)
        for k in range(n):
            minutes[mi, k] = int((assigned == k).sum())
        other[mi] = int((assigned == -1).sum())
        outside_m[mi] = int((blocked & (assigned == -2)).sum())

    explained = int(minutes.sum() + other.sum())  # the lost sun the photos can account for; "outside" is their limit
    items = []
    for k, g in enumerate(globals_):
        if minutes[:, k].max() < MIN_MONTH_HOURS * 60:
            continue
        lo, hi = g["arc"]
        items.append({
            "id": k, "kind": g["kind"], "name": KIND_NAMES.get(g["kind"], g["kind"].title()),
            "bearing": round((lo + _arc_len(g["arc"]) / 2.0) % 360.0, 1), "az_from": round(lo, 1), "az_to": round(hi, 1),
            "score": round(g["score"], 2),
            "hours": [round(m / 60.0, 2) for m in minutes[:, k]],
            "hours_year": round(float(minutes[:, k].mean()) / 60.0, 2),
            "share": round(float(minutes[:, k].sum()) / explained, 3) if explained else 0.0,
            "frames": g["frames"],
        })
    items.sort(key=lambda it: -it["hours_year"])
    order = {it["id"]: i for i, it in enumerate(items)}
    for it in items:
        it["id"] = order[it["id"]]
    for fr in frames:
        fr["global_ids"] = [order.get(g, -1) for g in fr["global_ids"]]
    return {
        "phrases": BLOCKER_PHRASES,
        "items": items,
        "other_hours": [round(m / 60.0, 2) for m in other],
        "outside_hours": [round(m / 60.0, 2) for m in outside_m],
        "blocked_hours": [round(m / 60.0, 2) for m in blocked_total],
    }


def _arc_len(arc: tuple[float, float]) -> float:
    lo, hi = arc
    return (hi - lo) % 360.0


def _union_arc(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """The clockwise arc covering two overlapping arcs."""
    a0, la = a[0] % 360.0, _arc_len(a)
    b0, lb = b[0] % 360.0, _arc_len(b)
    rel = (b0 - a0) % 360.0
    if rel <= la:  # b starts inside a
        return (a0, (a0 + max(la, rel + lb)) % 360.0)
    rel2 = (a0 - b0) % 360.0  # otherwise b starts before a and reaches into it
    return (b0, (b0 + max(lb, rel2 + la)) % 360.0)


def describe(blockers: dict) -> str:
    """The blockers as lines for a prompt or a terminal, with the months that matter."""
    if not blockers or not blockers.get("items") and not any(blockers.get("outside_hours", [])):
        return "No obstruction was named in the photos."
    lines = []
    for it in blockers["items"]:
        if max(it["hours"]) < 0.05:
            continue
        top = sorted(range(12), key=lambda i: -it["hours"][i])[:3]
        months = ", ".join(f"{MONTHS[i]} {it['hours'][i]:.1f} h" for i in sorted(top) if it["hours"][i] >= 0.05)
        lines.append(f"- {it['name']} at bearing {it['bearing']:.0f} ({it['az_from']:.0f} to {it['az_to']:.0f}): costs the "
                     f"spot {months} a day in those months, {it['hours_year']:.2f} h/day averaged over the year; "
                     f"{it['share'] * 100:.0f}% of the lost sun the photos can explain")
    other = float(np.mean(blockers.get("other_hours", [0])))
    out = float(np.mean(blockers.get("outside_hours", [0])))
    if other >= 0.05:
        lines.append(f"- Something unnamed on the skyline: {other:.1f} h/day on average")
    if out >= 0.05:
        lines.append(f"- Not an obstruction: {out:.1f} h/day falls in directions or heights the photos do not show, and is "
                     "counted as blocked only because nothing is known there (do not call this a wall or a building)")
    return "\n".join(lines) if lines else "Nothing in the photos costs the spot any direct sun."
