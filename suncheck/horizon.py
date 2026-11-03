"""The horizon of a spot: for each compass direction, how high the nearest obstruction (building, tree, overhang) rises.

Built from a photo's sky mask (SAM 3.1, noun phrase "sky") and the camera pose: the lowest sky pixel in each image column
gives one point of the horizon profile; several photos from the same spot merge into one profile over azimuth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraPose

AZ_STEP = 0.5
AZ_GRID = np.arange(0.0, 360.0, AZ_STEP)


@dataclass
class HorizonProfile:
    az: np.ndarray  # degrees, AZ_GRID
    el: np.ndarray  # obstruction elevation in degrees; NaN where no photo covers that azimuth

    def elevation_at(self, az_deg, unknown: float = 0.0) -> np.ndarray:
        """Obstruction elevation for arbitrary azimuths; directions no photo covers count as `unknown` degrees
        (0 for a flat open horizon, 90 to treat them as blocked, the honest choice for a balcony or window whose
        photos did not sweep the whole opening)."""
        az = np.asarray(az_deg, dtype=float) % 360.0
        idx = np.round(az / AZ_STEP).astype(int) % len(self.az)
        out = self.el[idx]
        return np.where(np.isnan(out), unknown, out)

    def coverage(self) -> tuple[float, float | None, float | None]:
        """Degrees of azimuth the profile knows, and the known range's start and end azimuths, clockwise (None when
        nothing is known). The range may cross north (a view from 293 to 60 reports from_az 293, to_az 60): the known
        arc is taken as everything outside the longest unknown arc of the circle."""
        known = ~np.isnan(self.el)
        n = len(known)
        if not known.any():
            return 0.0, None, None
        if known.all():
            return 360.0, 0.0, 360.0 - AZ_STEP
        first_known = int(np.argmax(known))
        rolled = np.roll(known, -first_known)  # starts on a known entry, so a gap that crosses north stays in one piece
        best_len, best_start, i = 0, 0, 0
        while i < n:
            if rolled[i]:
                i += 1
                continue
            j = i
            while j < n and not rolled[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        gap_start = (first_known + best_start) % n
        gap_end = (first_known + best_start + best_len) % n
        return float(known.sum() * AZ_STEP), float(self.az[gap_end]), float(self.az[(gap_start - 1) % n])

    def covered(self, az_deg) -> np.ndarray:
        az = np.asarray(az_deg, dtype=float) % 360.0
        idx = np.round(az / AZ_STEP).astype(int) % len(self.az)
        return ~np.isnan(self.el[idx])

    @classmethod
    def empty(cls) -> "HorizonProfile":
        return cls(AZ_GRID.copy(), np.full(len(AZ_GRID), np.nan))

    def merge(self, other: "HorizonProfile") -> "HorizonProfile":
        """Where both know a direction, keep the higher obstruction (the photo that saw the obstacle)."""
        el = np.where(np.isnan(self.el), other.el, np.where(np.isnan(other.el), self.el, np.maximum(self.el, other.el)))
        return HorizonProfile(self.az.copy(), el)


def skyline_from_mask(sky_mask: np.ndarray) -> np.ndarray:
    """For each image column, the row of the lowest sky pixel (the skyline), or -1 where the column has no sky."""
    h, w = sky_mask.shape
    has = sky_mask.any(axis=0)
    # lowest sky pixel: last True in each column
    rev = sky_mask[::-1, :]
    last = h - 1 - np.argmax(rev, axis=0)
    return np.where(has, last, -1)


def profile_from_photo(sky_mask: np.ndarray, pose: CameraPose, min_sky_fraction: float = 0.02) -> HorizonProfile:
    """Horizon profile over the azimuths this photo covers.

    Columns where sky reaches the bottom edge mean the obstruction is below the frame: the horizon there is at least as
    low as the frame's bottom edge, recorded as that elevation (a lower bound, flagged by being exactly the edge).
    """
    h, w = sky_mask.shape
    if sky_mask.mean() < min_sky_fraction:
        return HorizonProfile.empty()
    skyline = skyline_from_mask(sky_mask)
    cols = np.arange(w)
    valid = skyline >= 0
    rows = np.where(valid, skyline, h - 1).astype(float)
    az, el = pose.pixel_to_azel(cols.astype(float), rows)
    # columns without any sky: the obstruction fills the column; its top is at least the frame's top edge
    az_top, el_top = pose.pixel_to_azel(cols.astype(float), np.zeros(w))
    el = np.where(valid, el, el_top)
    prof = HorizonProfile.empty()
    idx = np.round((az % 360.0) / AZ_STEP).astype(int) % len(AZ_GRID)
    # several columns may land in one azimuth bin: keep the highest obstruction
    for i, e in zip(idx, el):
        if np.isnan(prof.el[i]) or e > prof.el[i]:
            prof.el[i] = e
    # fill tiny gaps between bins the photo spanned
    covered = ~np.isnan(prof.el)
    if covered.sum() >= 2:
        where = np.flatnonzero(covered)
        lo, hi = where.min(), where.max()
        span = np.arange(lo, hi + 1)
        wrap = (hi - lo) > len(AZ_GRID) / 2  # photo crosses north
        if not wrap:
            prof.el[span] = np.interp(span, where, prof.el[where])
    return prof


def sky_mask_fallback(image_rgb: np.ndarray) -> np.ndarray:
    """Sky without a model, for clear or lightly clouded skies: bright low-saturation or blue pixels connected to the top."""
    import cv2

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    blue = (h >= 90) & (h <= 135) & (s >= 25) & (v >= 90)
    bright = (v >= 200) & (s <= 60)
    cand = (blue | bright).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand, dtype=bool)
    top_rows = labels[: max(2, cand.shape[0] // 50), :]
    for lbl in np.unique(top_rows):
        if lbl != 0 and stats[lbl, cv2.CC_STAT_AREA] > cand.size * 0.002:
            keep |= labels == lbl
    return keep


def sun_pixel_from_mask(sun_mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(sun_mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def brightest_blob(gray: np.ndarray, percentile: float = 99.95) -> tuple[float, float] | None:
    """Fallback sun finder: centroid of the brightest saturated pixels."""
    thr = np.percentile(gray, percentile)
    ys, xs = np.nonzero(gray >= max(thr, 250))
    if len(xs) < 20:
        return None
    return float(xs.mean()), float(ys.mean())
