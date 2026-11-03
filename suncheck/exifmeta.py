"""What a phone photo already knows: where, when, which way it faced, how it was tilted, and its lens."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exifread


@dataclass
class PhotoMeta:
    path: Path
    width: int | None = None
    height: int | None = None
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    taken_local: datetime | None = None  # naive local time as written by the camera
    tz_offset_hours: float | None = None  # from OffsetTimeOriginal
    taken_utc: datetime | None = None  # from GPS time stamp when present, else from local + offset
    heading_deg: float | None = None  # GPSImgDirection
    heading_ref: str | None = None  # 'T' true north or 'M' magnetic
    f35_mm: float | None = None
    orientation: int | None = None
    acceleration: tuple[float, float, float] | None = None  # Apple maker note, units of g
    make_model: str = ""
    raw: dict = field(default_factory=dict, repr=False)


def _ratio(v) -> float:
    try:
        return float(v.num) / float(v.den)
    except AttributeError:
        return float(v)


def _dms(values, ref: str) -> float:
    d, m, s = (_ratio(x) for x in values)
    sign = -1.0 if ref.strip().upper() in ("S", "W") else 1.0
    return sign * (d + m / 60.0 + s / 3600.0)


def _pil_fallback(path: Path, meta: "PhotoMeta") -> None:
    """HEIC files that exifread cannot read: GPS, time, heading and lens through Pillow (pillow-heif registers HEIC)."""
    try:
        from PIL import Image

        if path.suffix.lower() in {".heic", ".heif"}:
            import pillow_heif

            pillow_heif.register_heif_opener()
        im = Image.open(path)
        exif = im.getexif()
        ifd = exif.get_ifd(0x8769)
        gps = exif.get_ifd(0x8825)
        meta.width, meta.height = im.size
        meta.make_model = f"{exif.get(0x010F, '')} {exif.get(0x0110, '')}".strip()
        meta.orientation = exif.get(0x0112)
        if ifd.get(0xA405):
            meta.f35_mm = float(ifd[0xA405])
        if isinstance(ifd.get(0x927C), bytes):
            meta.acceleration = apple_acceleration(ifd[0x927C])
        if ifd.get(0x9003):
            try:
                meta.taken_local = datetime.strptime(str(ifd[0x9003]), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        off = ifd.get(0x9011) or ifd.get(0x9010)
        if off:
            s = str(off).strip()
            sign = -1 if s.startswith("-") else 1
            hh, mm = s.lstrip("+-").split(":")
            meta.tz_offset_hours = sign * (int(hh) + int(mm) / 60.0)
        if gps.get(2) and gps.get(4):
            def dms(v, ref):
                d, m, s = (float(x) for x in v)
                return (-1 if str(ref).upper() in ("S", "W") else 1) * (d + m / 60 + s / 3600)
            meta.lat = dms(gps[2], gps.get(1, "N"))
            meta.lon = dms(gps[4], gps.get(3, "E"))
        if gps.get(17) is not None:
            meta.heading_deg = float(gps[17])
            meta.heading_ref = str(gps.get(16, "T")) or "T"
        if gps.get(6) is not None:
            meta.altitude_m = float(gps[6])
        if gps.get(29) and gps.get(7):
            try:
                y, mo, d = (int(x) for x in str(gps[29]).split(":"))
                h, mi, se = (float(x) for x in gps[7])
                meta.taken_utc = datetime(y, mo, d, int(h), int(mi), int(se), tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass
        if meta.taken_utc is None and meta.taken_local is not None and meta.tz_offset_hours is not None:
            meta.taken_utc = (meta.taken_local - timedelta(hours=meta.tz_offset_hours)).replace(tzinfo=timezone.utc)
    except Exception:
        pass


def read_meta(path: str | Path) -> PhotoMeta:
    path = Path(path)
    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=True)
    meta = PhotoMeta(path=path, raw={k: str(v) for k, v in tags.items() if not k.startswith("JPEGThumbnail")})
    if not tags:
        _pil_fallback(path, meta)
        return meta
    g = tags.get

    def num(key):
        t = g(key)
        return _ratio(t.values[0]) if t is not None and t.values else None

    if g("EXIF ExifImageWidth") and g("EXIF ExifImageLength"):
        meta.width, meta.height = int(num("EXIF ExifImageWidth")), int(num("EXIF ExifImageLength"))
    meta.make_model = f"{g('Image Make', '')} {g('Image Model', '')}".strip()
    meta.f35_mm = num("EXIF FocalLengthIn35mmFilm")
    meta.orientation = int(num("Image Orientation")) if g("Image Orientation") else None

    if g("GPS GPSLatitude") and g("GPS GPSLongitude"):
        meta.lat = _dms(g("GPS GPSLatitude").values, str(g("GPS GPSLatitudeRef", "N")))
        meta.lon = _dms(g("GPS GPSLongitude").values, str(g("GPS GPSLongitudeRef", "E")))
    if g("GPS GPSAltitude"):
        meta.altitude_m = num("GPS GPSAltitude")
    if g("GPS GPSImgDirection"):
        meta.heading_deg = num("GPS GPSImgDirection")
        meta.heading_ref = str(g("GPS GPSImgDirectionRef", "T")).strip() or "T"

    dto = g("EXIF DateTimeOriginal")
    if dto:
        try:
            meta.taken_local = datetime.strptime(str(dto), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    off = g("EXIF OffsetTimeOriginal") or g("EXIF OffsetTime")
    if off:
        s = str(off).strip()
        try:
            sign = -1 if s.startswith("-") else 1
            hh, mm = s.lstrip("+-").split(":")
            meta.tz_offset_hours = sign * (int(hh) + int(mm) / 60.0)
        except ValueError:
            pass
    if g("GPS GPSDateStamp") and g("GPS GPSTimeStamp"):
        try:
            y, mo, d = (int(x) for x in str(g("GPS GPSDateStamp")).split(":"))
            h, mi, se = (_ratio(x) for x in g("GPS GPSTimeStamp").values)
            meta.taken_utc = datetime(y, mo, d, int(h), int(mi), int(se), tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            pass
    if meta.taken_utc is None and meta.taken_local is not None and meta.tz_offset_hours is not None:
        meta.taken_utc = (meta.taken_local - timedelta(hours=meta.tz_offset_hours)).replace(tzinfo=timezone.utc)
    if meta.tz_offset_hours is None and meta.taken_utc is not None and meta.taken_local is not None:
        meta.tz_offset_hours = round((meta.taken_local - meta.taken_utc.replace(tzinfo=None)).total_seconds() / 3600.0, 2)

    # Apple maker note: AccelerationVector (tag 0x0008), three signed rationals in g, decoded from the raw bytes
    mn = tags.get("EXIF MakerNote")
    if mn is not None and meta.acceleration is None:
        try:
            meta.acceleration = apple_acceleration(bytes(mn.values))
        except (TypeError, ValueError):
            pass
    return meta


def apple_acceleration(maker_note: bytes) -> tuple[float, float, float] | None:
    """The AccelerationVector (tag 8) of an Apple maker note: three signed rationals in g.

    Apple's maker note is a little TIFF of its own whose value offsets count from the note's start; exifread resolves
    them against the file's TIFF header and returns nonsense for this tag (seen on iPhone 14 Pro originals, 3 November
    2026), so the entry is read here by hand.
    """
    import struct

    if not maker_note or not maker_note.startswith(b"Apple iOS") or len(maker_note) < 16:
        return None
    endian = ">" if maker_note[12:14] == b"MM" else "<"
    try:
        n = struct.unpack(endian + "H", maker_note[14:16])[0]
        for i in range(n):
            off = 16 + 12 * i
            tag, typ, cnt = struct.unpack(endian + "HHI", maker_note[off:off + 8])
            if tag == 8 and cnt == 3 and typ in (5, 10):
                valoff = struct.unpack(endian + "I", maker_note[off + 8:off + 12])[0]
                fmt = endian + ("ii" if typ == 10 else "II")
                vals = []
                for j in range(3):
                    num, den = struct.unpack(fmt, maker_note[valoff + 8 * j:valoff + 8 * j + 8])
                    vals.append(num / den if den else 0.0)
                return (vals[0], vals[1], vals[2])
    except struct.error:
        return None
    return None


def pitch_roll_from_acceleration(acc: tuple[float, float, float], orientation: int | None):
    """Camera pitch (elevation of the optical axis) and roll from Apple's acceleration vector.

    ExifTool documents the vector as viewed from the front of the phone: +X toward the left side, +Y toward the bottom,
    +Z into the face of the phone (units of g); the reading is the reaction, so at rest it points up, away from gravity.
    The back camera therefore looks along +Z (into the phone, out of its back). Verified on 3 November 2026 against
    four iPhone 14 Pro originals: the horizon sat 14 to 34 degrees above the frame centre and the pitch came out at
    -14 to -34 degrees; the frames were upright and the roll came out within a few degrees of zero.

    Roll is the tilt of the image's horizontal axis after the EXIF orientation is applied: for the usual portrait hold
    (orientation 6) that axis is the phone's own X; upside down (8) it is reversed; the two landscape holds (1 and 3)
    use the phone's Y axis, with the sign of that shift not yet checked on a landscape original.
    """
    import math

    x, y, z = acc
    gx, gy, gz = -x, -y, -z  # gravity in the phone frame
    norm = math.sqrt(gx * gx + gy * gy + gz * gz) or 1.0
    gx, gy, gz = gx / norm, gy / norm, gz / norm
    pitch = math.degrees(math.asin(-gz))  # optical axis (0, 0, +1): above the horizontal when gravity has no +Z part
    roll = math.degrees(math.atan2(gx, gy))  # portrait: image x along phone X, gravity sits along +Y when level (sign of roll unverified)
    if orientation == 8:
        roll += 180.0
    elif orientation in (1, 3):  # landscape holds: the image x axis is the phone's Y axis (shift sign unverified)
        roll += 90.0 if orientation == 1 else -90.0
    roll = (roll + 180.0) % 360.0 - 180.0
    return pitch, roll


def dump_tags(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=True)
    return {k: str(v)[:80] for k, v in tags.items() if not k.startswith("JPEGThumbnail")}
