"""Synthetic checks that need no network: a known skyline rendered through a known camera, then recovered.

1. Horizon: a true obstruction profile h(az) is rendered into a sky mask with the forward camera model; profile_from_photo
   must give it back over the covered azimuths.
2. Sun calibration: the sun at a known time is projected into the frame; solve_yaw_pitch_from_sun must recover the pose.
3. Hours: the monthly table from the recovered profile must match the table from the true profile.
4. A drawn overlay for eyeballing.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraPose, f_px_from_f35, solve_yaw_pitch_from_sun
from .horizon import AZ_GRID, HorizonProfile, profile_from_photo
from .hours import draw_overlay, monthly_table
from .solar import sun_position, utc_from_local

LAT, LON, TZ = 25.2048, 55.2708, 4.0  # Dubai


def true_horizon(az_deg: np.ndarray) -> np.ndarray:
    """A made-up skyline: blocks of buildings and a tree, elevation in degrees by azimuth."""
    az = np.asarray(az_deg) % 360.0
    el = np.full(az.shape, 1.5)  # distant flat horizon slightly above zero (haze, far buildings)
    for a0, a1, e in ((150, 170, 12.0), (172, 185, 22.0), (190, 205, 8.0), (215, 232, 17.0), (240, 250, 6.0)):
        el = np.where((az >= a0) & (az <= a1), e, el)
    tree = np.exp(-((az - 208.0) / 3.0) ** 2) * 14.0
    return np.maximum(el, tree)


def render_mask(pose: CameraPose, step: int = 1) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(0, pose.width, step), np.arange(0, pose.height, step))
    az, el = pose.pixel_to_azel(us.astype(float), vs.astype(float))
    sky = el > true_horizon(az)
    if step > 1:
        sky = cv2.resize(sky.astype(np.uint8), (pose.width, pose.height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return sky


def run(out_dir: Path, width: int = 4032, height: int = 3024, f35: float = 24.0, yaw: float = 225.0, pitch: float = 10.0,
        roll: float = 0.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    f_px = f_px_from_f35(f35, width, height)
    truth = CameraPose(f_px, width, height, yaw, pitch, roll)
    results = {"f_px": f_px, "hfov_deg": truth.hfov_deg(), "vfov_deg": truth.vfov_deg()}

    # 1. horizon recovery (render at reduced resolution for speed, upsample the mask)
    mask = render_mask(truth, step=4)
    prof = profile_from_photo(mask, truth)
    covered = ~np.isnan(prof.el)
    err = np.abs(prof.el[covered] - true_horizon(AZ_GRID[covered]))
    # bins at the image's left and right edges are partial; ignore the outermost two
    inner = covered.copy()
    idx = np.flatnonzero(covered)
    inner[idx[:2]] = False
    inner[idx[-2:]] = False
    err_inner = np.abs(prof.el[inner] - true_horizon(AZ_GRID[inner]))
    results["horizon"] = {
        "covered_azimuth_span_deg": float(covered.sum() * (AZ_GRID[1] - AZ_GRID[0])),
        "max_error_deg": float(err_inner.max()),
        "p95_error_deg": float(np.percentile(err_inner, 95)),
        "median_error_deg": float(np.median(err_inner)),
        "mean_error_deg": float(err_inner.mean()),
        "note": "large errors sit in the half-degree bins that straddle a vertical building edge, where the bin keeps the higher obstruction",
    }

    # 2. sun calibration
    t_local = datetime(2026, 11, 2, 16, 30)  # sun low enough (about 15 degrees) to sit inside a level-ish frame
    az_s, el_s = sun_position(LAT, LON, np.datetime64(utc_from_local(t_local, TZ).replace(tzinfo=None)))
    az_s, el_s = float(az_s), float(el_s)
    u, v, vis = truth.azel_to_pixel(np.array([az_s]), np.array([el_s]))
    results["sun"] = {"az": az_s, "el": el_s, "pixel": [float(u[0]), float(v[0])], "visible": bool(vis[0])}
    if vis[0]:
        solved = solve_yaw_pitch_from_sun(f_px, width, height, (float(u[0]), float(v[0])), az_s, el_s, roll=roll,
                                          yaw0=yaw + 25, pitch0=0.0)
        results["sun"]["solved_yaw"] = solved.yaw
        results["sun"]["solved_pitch"] = solved.pitch
        results["sun"]["yaw_error_deg"] = float((solved.yaw - yaw + 180) % 360 - 180)
        results["sun"]["pitch_error_deg"] = float(solved.pitch - pitch)
        # sensitivity: solving with roll assumed zero when the true roll was 3 degrees
        tilted = CameraPose(f_px, width, height, yaw, pitch, 3.0)
        u3, v3, _ = tilted.azel_to_pixel(np.array([az_s]), np.array([el_s]))
        wrong = solve_yaw_pitch_from_sun(f_px, width, height, (float(u3[0]), float(v3[0])), az_s, el_s, roll=0.0, yaw0=yaw, pitch0=pitch)
        results["sun"]["error_if_3deg_roll_ignored"] = {"yaw": float((wrong.yaw - yaw + 180) % 360 - 180), "pitch": float(wrong.pitch - pitch)}

    # 3. hours from the recovered profile against the true profile
    true_prof = HorizonProfile(AZ_GRID.copy(), true_horizon(AZ_GRID))
    rec_full = HorizonProfile(AZ_GRID.copy(), np.where(covered, prof.el, true_prof.el))  # outside the photo, assume known
    t_true = monthly_table(LAT, LON, 2026, TZ, true_prof)
    t_rec = monthly_table(LAT, LON, 2026, TZ, rec_full)
    results["hours"] = [{"month": a["month"], "true": a["direct_sun_hours"], "recovered": b["direct_sun_hours"],
                         "flat_horizon": None} for a, b in zip(t_true, t_rec)]
    flat = monthly_table(LAT, LON, 2026, TZ, HorizonProfile.empty())
    for row, f in zip(results["hours"], flat):
        row["flat_horizon"] = f["direct_sun_hours"]
    results["hours_max_abs_diff"] = float(max(abs(r["true"] - r["recovered"]) for r in results["hours"]))

    # 4. overlay on a synthetic image: sky gradient above the skyline, grey blocks below
    img = np.zeros((height, width, 3), np.uint8)
    img[mask] = (235, 206, 135)  # sky (BGR light blue)
    img[~mask] = (90, 90, 90)
    small = cv2.resize(img, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    small_pose = CameraPose(f_px / 4, width // 4, height // 4, yaw, pitch, roll)
    over = draw_overlay(small, small_pose, LAT, LON, 2026, TZ, prof, sun_now=(az_s, el_s), label_hours=t_rec)
    cv2.imwrite(str(out_dir / "selftest_overlay.png"), over)
    (out_dir / "selftest.json").write_text(json.dumps(results, indent=2, default=str))
    return results
