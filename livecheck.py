"""End-to-end check of a running Suncheck server: post frames to /api/analyze, poll the job, fetch the assets, ask Spark.

  python3 livecheck.py BASE_URL --sample            # upload mode: the shipped sample frame, place and time given
  python3 livecheck.py BASE_URL --job out/web/<id>  # camera mode: a finished job's frames and recorded sensors
"""
import json, sys, time, uuid, argparse
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--sample", action="store_true")
ap.add_argument("--job")
ap.add_argument("--files", nargs="+", help="upload mode: camera-app originals with EXIF (HEIC or JPEG)")
ap.add_argument("--question", default="Is this spot good for tomatoes?")
ap.add_argument("--project", default=str(Path(__file__).resolve().parent))
a = ap.parse_args()
base = a.base.rstrip("/")
proj = Path(a.project)

def get(path, binary=False):
    with urlopen(Request(base + path, headers={"User-Agent": "livecheck"}), timeout=120) as r:
        data = r.read()
        return (r.status, data if binary else data.decode())

def post_multipart(path, fields, files):
    bnd = "----suncheck" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for name, fname, payload, ctype in files:
        body += f"--{bnd}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode() + payload + b"\r\n"
    body += f"--{bnd}--\r\n".encode()
    req = Request(base + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={bnd}", "User-Agent": "livecheck"})
    with urlopen(req, timeout=300) as r:
        return r.status, json.loads(r.read())

def post_json(path, obj):
    req = Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json", "User-Agent": "livecheck"})
    with urlopen(req, timeout=300) as r:
        return r.status, json.loads(r.read())

t0 = time.time()
print("health:", get("/api/health")[1])
if a.files:
    meta = {"mode": "upload", "spot": "balcony", "advice": True}
    files = [("photos", Path(f).name, Path(f).read_bytes(), "image/heic" if f.lower().endswith(".heic") else "image/jpeg") for f in a.files]
elif a.sample:
    frame = proj / "suncheck/sample/sample_frame.jpg"
    meta = {"mode": "upload", "lat": 25.2, "lon": 55.27, "tz_min": 240, "taken": "2026-11-03 16:45", "spot": "balcony", "advice": True}
    files = [("photos", "sample_frame.jpg", frame.read_bytes(), "image/jpeg")]
else:
    jd = Path(a.job)
    rep = json.loads((jd / "report.json").read_text())
    shots, files = [], []
    ua = None
    for ph in rep["photos"]:
        s = dict(ph.get("sensors") or {})
        ua = s.pop("ua", ua)
        t_ms = int(datetime.fromisoformat(ph["taken_utc"]).timestamp() * 1000)
        shots.append({"lat": ph.get("lat"), "lon": ph.get("lon"), "t_ms": t_ms, **{k: s.get(k) for k in ("alpha", "beta", "gamma", "absolute", "compass", "compass_acc", "angle", "acc_m", "alt", "w", "h")}})
        files.append(("photos", ph["file"], (jd / "in" / ph["file"]).read_bytes(), "image/jpeg"))
    meta = {"mode": "camera", "tz_min": 240, "ua": ua, "photos": shots, "spot": "balcony", "advice": True}
print(f"posting {len(files)} file(s), {sum(len(f[2]) for f in files)/1e6:.1f} MB")
st, resp = post_multipart("/api/analyze", {"meta": json.dumps(meta)}, files)
print("analyze:", st, resp, f"({time.time()-t0:.1f}s)")
job = resp["job"]
last = None
while True:
    st, body = get(f"/api/job/{job}")
    snap = json.loads(body)
    line = f"{snap.get('state')} {snap.get('progress'):.2f} {snap.get('message')}"
    if line != last:
        print(f"  {time.time()-t0:6.1f}s  {line}")
        last = line
    if snap.get("state") in ("done", "error"):
        break
    if time.time() - t0 > 900:
        print("TIMEOUT"); sys.exit(1)
    time.sleep(2)
if snap["state"] == "error":
    print("ERROR:", snap.get("error")); sys.exit(1)
rep = snap["result"]
print("summary:", rep["summary"]["headline"], "|", rep["summary"]["subline"])
print("coverage:", rep["coverage"])
print("sun check:", rep.get("sun_check_summary"))
for ph in rep["photos"]:
    print(f"  {ph['file']}: pose {ph['pose_used']['source']}, sun {ph['sun_detection']['source']}, check {json.dumps(ph.get('sun_check'))[:200]}")
print("hours:", ", ".join(f"{m['month']} {m['direct_sun_hours']}" for m in rep["monthly_direct_sun_hours"]))
adv = rep.get("advice") or {}
print("advice:", json.dumps(adv)[:300])
for key in ("overlay_url", "overlay_all_url", "sky_url", "frame_url"):
    u = rep["photos"][0][key]
    st, data = get(u, binary=True)
    print(f"  asset {u}: {st} {len(data)} bytes")
if rep.get("card_url"):
    st, data = get(rep["card_url"], binary=True)
    print(f"  card {rep['card_url']}: {st} {len(data)} bytes")
st, ans = post_json("/api/ask", {"job": job, "question": a.question})
print("ask:", st, json.dumps(ans)[:300])
print(f"result page: {base}/?job={job}   total {time.time()-t0:.1f}s")
