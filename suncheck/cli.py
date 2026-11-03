from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(args) -> dict:
    from .pipeline import Options, analyze_set, photo_from_file

    sun_pixel = tuple(float(x) for x in args.sun_pixel.split(",")) if args.sun_pixel else None
    photos = [photo_from_file(Path(p), lat=args.lat, lon=args.lon, tz=args.tz, taken_local=args.taken, f35=args.f35,
                              yaw=args.yaw, pitch=args.pitch, roll=args.roll, declination=args.declination, sun_pixel=sun_pixel)
              for p in args.images]
    opts = Options(outside=args.outside, use_sam=not args.no_sam, pose=args.pose, advice=args.advice)
    return analyze_set(photos, opts, Path(args.out), progress=lambda t, f, **_: print(f"  [{f:4.0%}] {t}"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="suncheck", description="How much sun does this spot really get.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="photos from one spot in, sun paths and monthly hours out")
    a.add_argument("images", nargs="+")
    a.add_argument("--out", default="out/analysis")
    a.add_argument("--lat", type=float)
    a.add_argument("--lon", type=float)
    a.add_argument("--tz", type=float, help="UTC offset in hours if EXIF lacks it")
    a.add_argument("--f35", type=float)
    a.add_argument("--taken", help="local capture time 'YYYY-MM-DD HH:MM' when EXIF lacks it (photos sent through chat apps lose it)")
    a.add_argument("--yaw", type=float, help="override compass heading of the optical axis")
    a.add_argument("--pitch", type=float)
    a.add_argument("--roll", type=float)
    a.add_argument("--declination", type=float, default=0.0, help="magnetic declination to add when EXIF heading is magnetic")
    a.add_argument("--sun-pixel", help="u,v of the sun in the image, if automatic detection fails")
    a.add_argument("--pose", choices=["sensors", "sun"], default="sun", help="which pose to use when the sun is in frame")
    a.add_argument("--no-sam", action="store_true")
    a.add_argument("--outside", choices=["blocked", "open"], default="blocked",
                   help="directions the photos do not cover: blocked (balcony, window) or open flat horizon (garden)")
    a.add_argument("--advice", action="store_true", help="ask Muse Spark to word the verdict")
    s = sub.add_parser("selftest")
    s.add_argument("--out", default="out/selftest")
    w = sub.add_parser("web", help="serve the capture page")
    w.add_argument("--host", default="0.0.0.0")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--cert")
    w.add_argument("--key")
    args = p.parse_args(argv)
    if args.cmd == "selftest":
        from .selftest import run

        print(json.dumps(run(Path(args.out)), indent=1, default=str)[:3000])
    elif args.cmd == "web":
        from .web import main as web_main

        web_main(["--host", args.host, "--port", str(args.port)] + (["--cert", args.cert, "--key", args.key] if args.cert else []))
    else:
        rep = analyze(args)
        for ph in rep["photos"]:
            line = f"{ph['file']}: heading from {ph['heading_source']} | sun az {ph['sun_now']['az']:.1f} el {ph['sun_now']['el']:.1f}"
            if ph["sun_check"]:
                line += f" | sun check error {ph['sun_check']['angular_error_deg']:.1f} deg"
            if ph["pose_from_sun"]:
                s = ph["pose_from_sun"]
                line += f" | pose from sun: yaw {s['yaw']:.1f} pitch {s['pitch']:.1f}" + (f" f {s['f_px']:.0f}px" if s["lens_solved"] else "")
            line += f" | sky {ph['sky_fraction']:.0%} via {ph['sky_source']}"
            print(line)
        c = rep["coverage"]
        print(f"photos cover {c['azimuth_deg']:.0f} deg of azimuth ({c['from_az']} to {c['to_az']}); outside that: {c['outside']}")
        print("month  direct-sun hours  first  last")
        for r in rep["monthly_direct_sun_hours"]:
            print(f"{r['month']:>5} {r['direct_sun_hours']:8.2f}  {r['first_sun']} {r['last_sun']}")
        for n in rep["notes"]:
            print("note:", n)
        if rep.get("advice"):
            print()
            print(rep["advice"])
