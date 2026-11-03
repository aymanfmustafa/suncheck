"""Sun position and daily paths: the NOAA general solar position algorithm, vectorised over time.

Angles in degrees. Azimuth is clockwise from true north; elevation is geometric (no refraction) unless asked for.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np


def _julian_day(dt_utc: np.ndarray) -> np.ndarray:
    """Julian day from an array of numpy datetime64[s] in UTC."""
    unix = dt_utc.astype("datetime64[s]").astype(np.int64)
    return unix / 86400.0 + 2440587.5


def sun_position(lat: float, lon: float, dt_utc) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth and geometric elevation of the sun for UTC datetimes (scalar, list or datetime64 array)."""
    dt = np.asarray(dt_utc, dtype="datetime64[s]")
    jd = _julian_day(dt)
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + np.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + np.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = np.radians(125.04 - 1934.136 * T)
    app_long = true_long - 0.00569 - 0.00478 * np.sin(omega)
    eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0) / 60.0
    eps = np.radians(eps0 + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(eps) * np.sin(np.radians(app_long)))
    y = np.tan(eps / 2) ** 2
    L0r = np.radians(L0)
    eot = 4.0 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                           - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    minutes_utc = (dt.astype("datetime64[s]").astype(np.int64) % 86400) / 60.0
    tst = (minutes_utc + eot + 4.0 * lon) % 1440.0
    H = np.radians(tst / 4.0 - 180.0)
    phi = np.radians(lat)
    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(H)
    cos_z = np.clip(cos_z, -1.0, 1.0)
    elevation = 90.0 - np.degrees(np.arccos(cos_z))
    # azimuth clockwise from north: atan2 form, hour angle positive after local noon (west)
    az = np.degrees(np.arctan2(np.sin(H), np.cos(H) * np.sin(phi) - np.tan(decl) * np.cos(phi))) + 180.0
    return az % 360.0, elevation


def refraction_correction(elevation_deg: np.ndarray) -> np.ndarray:
    """Approximate atmospheric refraction in degrees (NOAA), to add to the geometric elevation for the apparent sun."""
    el = np.asarray(elevation_deg, dtype=float)
    r = np.zeros_like(el)
    te = np.tan(np.radians(np.clip(el, -5, 90)))
    high = el > 85
    mid = (el > 5) & ~high
    low = (el > -0.575) & ~mid & ~high
    r[mid] = (58.1 / te[mid] - 0.07 / te[mid] ** 3 + 0.000086 / te[mid] ** 5) / 3600.0
    r[low] = (1735 + el[low] * (-518.2 + el[low] * (103.4 + el[low] * (-12.79 + el[low] * 0.711)))) / 3600.0
    rest = ~(high | mid | low)
    r[rest] = (-20.774 / np.tan(np.radians(np.where(rest, el, -5)))[rest]) / 3600.0
    return r


def day_path(lat: float, lon: float, day: date, tz_offset_hours: float, step_minutes: int = 1):
    """Sun azimuth and elevation through one local calendar day. Returns (local_times, az, el)."""
    start_local = datetime.combine(day, time(0, 0))
    n = int(24 * 60 / step_minutes)
    locals_ = np.array([start_local + timedelta(minutes=step_minutes * i) for i in range(n)], dtype="datetime64[s]")
    utc = locals_ - np.timedelta64(int(round(tz_offset_hours * 3600)), "s")
    az, el = sun_position(lat, lon, utc)
    return locals_, az, el


def month_days(year: int, day_of_month: int = 21) -> list[date]:
    return [date(year, m, day_of_month) for m in range(1, 13)]


def utc_from_local(dt_local: datetime, tz_offset_hours: float) -> datetime:
    return (dt_local - timedelta(hours=tz_offset_hours)).replace(tzinfo=timezone.utc)
