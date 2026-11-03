"""The capture page and its API: the phone takes the frames and reads its own sensors; this server runs the pipeline.

Standard library only (ThreadingHTTPServer), so it runs wherever the package does: `python -m suncheck.web --port 8000`.
Phones need a secure context for the camera and the motion sensors, so put it behind HTTPS (a tunnel, or --cert/--key).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import ssl
import threading
import traceback
import uuid
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .camera import pose_from_device_orientation
from .pipeline import Options, PhotoInput, analyze_set, photo_from_file

STATIC = Path(__file__).resolve().parent / "static"
SAMPLE = Path(__file__).resolve().parent / "sample"  # a finished result shipped with the app, for visitors without a phone
OUT = Path(__file__).resolve().parent.parent / "out" / "web"
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
MAX_BODY = 200 * 1024 * 1024


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, str, bytes]]]:
    """Form fields and files from a multipart/form-data body, through the standard library's MIME parser."""
    head = b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n"
    msg = BytesParser(policy=HTTP).parsebytes(head + body)
    fields: dict[str, str] = {}
    files: list[tuple[str, str, bytes]] = []
    if not msg.is_multipart():
        return fields, files
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append((name, filename, payload))
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


def _num(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def photos_from_request(meta: dict, files: list[tuple[str, str, bytes]], job_dir: Path) -> list[PhotoInput]:
    """PhotoInputs from the page's JSON and files: camera frames with sensor readings, or uploaded photos with EXIF."""
    in_dir = job_dir / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    tz = _num(meta.get("tz_min"))
    tz = tz / 60.0 if tz is not None else None
    photos: list[PhotoInput] = []
    if meta.get("mode") == "camera":
        shots = meta.get("photos") or []
        for i, (_, filename, payload) in enumerate(files):
            m = shots[i] if i < len(shots) else {}
            safe = f"shot_{i + 1}.jpg"
            path = in_dir / safe
            path.write_bytes(payload)
            p = PhotoInput(path=path, name=safe, lat=_num(m.get("lat")), lon=_num(m.get("lon")), tz=tz,
                           f35=_num(m.get("f35")) or None, sensors={k: m.get(k) for k in
                           ("alpha", "beta", "gamma", "absolute", "compass", "compass_acc", "angle", "acc_m", "alt", "w", "h")})
            p.sensors["ua"] = meta.get("ua")
            t_ms = _num(m.get("t_ms"))
            if t_ms:
                p.taken_utc, p.time_source = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc), "phone clock"
            beta, gamma, alpha, compass = _num(m.get("beta")), _num(m.get("gamma")), _num(m.get("alpha")), _num(m.get("compass"))
            angle = _num(m.get("angle")) or 0.0
            if beta is not None and gamma is not None:
                if compass is not None:
                    alpha_abs, p.heading_source = (360.0 - compass) % 360.0, "phone compass (webkitCompassHeading)"
                elif alpha is not None and m.get("absolute"):
                    alpha_abs, p.heading_source = alpha, "phone compass (absolute alpha)"
                else:
                    alpha_abs = None
                if alpha_abs is not None:
                    p.yaw, p.pitch, p.roll = pose_from_device_orientation(alpha_abs, beta, gamma, angle)
                else:
                    _, p.pitch, p.roll = pose_from_device_orientation(0.0, beta, gamma, angle)
                    p.heading_source = "none (no compass reading)"
                p.tilt_trusted = True
            photos.append(p)
    else:
        for i, (_, filename, payload) in enumerate(files):
            suffix = Path(filename).suffix.lower() or ".jpg"
            path = in_dir / f"upload_{i + 1}{suffix}"
            path.write_bytes(payload)
            p = photo_from_file(path, lat=_num(meta.get("lat")), lon=_num(meta.get("lon")), tz=tz,
                                taken_local=(meta.get("taken") or None), f35=_num(meta.get("f35")), yaw=_num(meta.get("heading")),
                                pitch=_num(meta.get("pitch")), roll=_num(meta.get("roll")))
            p.name = Path(filename).name
            photos.append(p)
    return photos


def _with_urls(rep: dict, job_id: str) -> dict:
    for ph in rep["photos"]:
        ph["overlay_url"] = f"/results/{job_id}/{ph['overlay']}"
        ph["overlay_all_url"] = f"/results/{job_id}/{ph['overlay_all']}"
        ph["sky_url"] = f"/results/{job_id}/{ph['sky_image']}"
        ph["frame_url"] = f"/results/{job_id}/{ph['frame']}"
        ph["sky_png_url"] = f"/results/{job_id}/{ph['sky_png']}"
    if rep.get("card"):
        rep["card_url"] = f"/results/{job_id}/{rep['card']}"
    return rep


def load_finished(job_id: str) -> dict | None:
    """A result from an earlier run of the server: report.json on disk, so result links outlive restarts."""
    if not job_id.isalnum():
        return None
    path = OUT / job_id / "report.json"
    if not path.is_file() and job_id == "sample":
        path = SAMPLE / "report.json"
    if not path.is_file():
        return None
    rep = _with_urls(json.loads(path.read_text()), job_id)
    with LOCK:
        JOBS[job_id] = {"id": job_id, "state": "done", "progress": 1.0, "message": "Done", "photos": len(rep["photos"]), "result": rep}
        return dict(JOBS[job_id])


def run_job(job_id: str, photos: list[PhotoInput], opts: Options, job_dir: Path) -> None:
    def progress(text: str, frac: float, partial=None, **_):
        with LOCK:
            JOBS[job_id].update(message=text, progress=frac)
            if partial is not None:
                for fr in partial:
                    fr["frame_url"] = f"/results/{job_id}/{fr['frame']}"
                JOBS[job_id]["partial"] = partial

    try:
        with LOCK:
            JOBS[job_id].update(state="running")
        rep = _with_urls(analyze_set(photos, opts, job_dir, progress), job_id)
        with LOCK:
            JOBS[job_id].update(state="done", result=rep, progress=1.0, message="Done")
    except Exception as e:  # noqa: BLE001
        with LOCK:
            JOBS[job_id].update(state="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())


class Handler(BaseHTTPRequestHandler):
    server_version = "suncheck/0.1"

    def log_message(self, fmt, *args):  # quieter log: one line per request
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, ctype: str = "application/json", cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode())

    def _file(self, base: Path, rel: str, cache: bool = False) -> None:
        target = (base / rel).resolve()
        if not str(target).startswith(str(base.resolve())) or not target.is_file():
            self._json(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype, cache)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(STATIC, "index.html")
        elif path.startswith("/static/"):
            self._file(STATIC, path[len("/static/"):], cache=True)
        elif path.startswith("/results/sample/") and not (OUT / "sample").is_dir():
            self._file(SAMPLE, path[len("/results/sample/"):], cache=True)
        elif path.startswith("/results/"):
            self._file(OUT, path[len("/results/"):], cache=True)
        elif path == "/favicon.ico":
            self._send(200, b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#000"/><circle cx="16" cy="16" r="7" fill="#F5B342"/></svg>', "image/svg+xml", cache=True)
        elif path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            with LOCK:
                job = JOBS.get(job_id)
                snap = dict(job) if job else None
            if not snap:
                snap = load_finished(job_id)
            if not snap:
                self._json(404, {"error": "no such job"})
            else:
                snap.pop("trace", None)
                self._json(200, snap)
        elif path == "/api/health":
            self._json(200, {"ok": True, "jobs": len(JOBS)})
        else:
            self._json(404, {"error": "not found"})

    def do_HEAD(self):  # noqa: N802  (tunnel services probe with HEAD)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": "body missing or too large"})
            return
        body = self.rfile.read(length)
        if len(body) < length:
            self._json(400, {"error": f"upload incomplete ({len(body)} of {length} bytes); try again"})
            return
        if path == "/api/ask":
            try:
                q = json.loads(body.decode("utf-8", "replace"))
                with LOCK:
                    job = JOBS.get(str(q.get("job")))
                    rep = job.get("result") if job else None
                if not rep:
                    found = load_finished(str(q.get("job")))
                    rep = found.get("result") if found else None
                if not rep:
                    self._json(404, {"error": "no such result"})
                    return
                question = str(q.get("question") or "").strip()[:400]
                if not question:
                    self._json(400, {"error": "empty question"})
                    return
                from .advice import answer_question

                self._json(200, {"answer": answer_question(rep, question)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})
            return
        if path != "/api/analyze":
            self._json(404, {"error": "not found"})
            return
        ctype = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in ctype:
            self._json(400, {"error": "multipart/form-data expected"})
            return
        fields, files = parse_multipart(ctype, body)
        try:
            meta = json.loads(fields.get("meta") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "meta is not JSON"})
            return
        if not files:
            self._json(400, {"error": "no photos"})
            return
        job_id = uuid.uuid4().hex[:12]
        job_dir = OUT / job_id
        try:
            photos = photos_from_request(meta, files, job_dir)
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"{type(e).__name__}: {e}"})
            return
        spot = str(meta.get("spot") or "balcony")
        outside = meta.get("outside") or ("open" if spot in ("garden", "rooftop") else "blocked")
        opts = Options(outside=outside, advice=bool(meta.get("advice", True)), use_sam=not meta.get("no_sam", False))
        with LOCK:
            JOBS[job_id] = {"id": job_id, "state": "queued", "progress": 0.0, "message": "Queued", "photos": len(photos),
                            "created": datetime.now(timezone.utc).isoformat()}
        threading.Thread(target=run_job, args=(job_id, photos, opts, job_dir), daemon=True).start()
        self._json(202, {"job": job_id, "photos": len(photos)})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="suncheck.web", description="Serve the Suncheck capture page.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cert", help="PEM certificate for HTTPS")
    ap.add_argument("--key", help="PEM private key for HTTPS")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    scheme = "http"
    if args.cert and args.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"Suncheck page on {scheme}://{args.host}:{args.port}/  (results in {OUT})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
