# Suncheck

How much sun does this spot really get? Photograph the view from a balcony, a window or a garden corner; Suncheck traces
the real skyline, checks its own maths against the sun in your frame, and shows you the sun crossing your own sky for
every month of the year, with the hours of direct sun each month and a plain-English verdict.

Built from scratch for the Meta Global AI Developer Hackathon (27 October to 4 November 2026) by Ayman with Claude Code.
Everything in this folder was written during the hackathon.

## What does the work

- **SAM 3.1** (Meta Model API) segments the sky and finds the sun in each photo.
- **Our code** does the measurement: solar position (NOAA algorithm, checked against `astral` to 0.46°), a pinhole camera
  model with the phone's tilt from its accelerometer, the heading and the lens solved from the sun's position in the frame,
  the horizon profile from the sky mask (one obstruction elevation per half degree of azimuth), and the minute-by-minute
  count of direct sun on the 21st of each month.
- **Muse Spark 1.3** words the result (what blocks the sun, good for, not for, one tip) from our numbers and answers the
  user's questions about the spot with the numbers in context.
- **Muse Image** generated the sample view shipped with the app (`suncheck/sample/`), which then went through the same
  pipeline as a real photo; the app says so.

The phone's compass is never trusted on its own: wherever the sun is in a frame, the sun fixes that frame's heading, and
the difference to the compass is applied to the frames without the sun. The hours count only the directions the photos
cover; the rest is always treated as blocked and the result says how much of the horizon was covered. No sky in any
frame gives "No sky in these photos", never a sunny answer.

## Run it

```bash
uv sync
printf 'META_API_KEY=...\n' > .env            # git-ignored
uv run python -m suncheck selftest            # synthetic skyline: geometry round trip, no API calls
uv run python -m suncheck analyze photos/*.HEIC --out out/spot --advice
uv run python -m suncheck web --port 8000     # the page, standard library only; results in out/web/<job>/
python livecheck.py http://localhost:8000 --sample   # end-to-end check of a running server, local or live
```

The page needs HTTPS on a phone (camera, compass, location). For a quick test: `npx localtunnel --port 8000`; the first
visit asks for the machine's public IP as a password (`curl https://loca.lt/mytunnelpassword`). The tunnel must be
restarted whenever the server restarts. `/?job=sample` shows the shipped sample; `/?job=<id>` reopens any result.

**Live link (deployed 3 November):** https://suncheck-447203242444.me-central1.run.app. It runs on Google Cloud Run (project `tifyal`, region `me-central1`, Doha)
from the `Dockerfile` here, built remotely by `gcloud run deploy suncheck --source . --region me-central1` with the key
passed as `--set-env-vars META_API_KEY=...`. The settings that matter: `--no-cpu-throttling`, because the analysis runs in
a background thread after the upload returns and the CPU would otherwise stop between the page's polls;
`--min-instances 1 --max-instances 1`, so the running job's progress stays in one instance's memory and there is no cold
start; 2 CPUs and 2 GiB for the 4K frames. Results in `out/web/` live as long as the instance (a redeploy clears them);
the sample is inside the package, so `/?job=sample` always works. Redeploy with the same command after a change. The
always-on instance costs about a dollar a day; `gcloud run services update suncheck --region me-central1 --min-instances 0`
lets it sleep (results vanish when it does) and `gcloud run services delete suncheck --region me-central1` removes it.

## Layout

`suncheck/solar.py` sun position and day paths · `camera.py` pinhole model, pose from the sun, pose from the phone's
orientation angles · `exifmeta.py` EXIF (GPS, time, heading, lens) · `horizon.py` sky mask to horizon profile ·
`hours.py` monthly table and the labelled overlay · `pipeline.py` the whole chain for a set of photos (`analyze_set`),
the words of the answer, the frame assets the page animates · `sam.py` SAM 3.1 client and mask decoding ·
`advice.py` Spark card and questions · `card.py` the share card · `web.py` the job API and static server ·
`static/index.html` the page · `sample/` the shipped sample result · `cli.py` the command line.

## The page

One page, four screens, phone first:

1. **Opening.** The sample view with the result's instruments drawn on it; "Open the camera" (phones), "See a sample",
   "Upload photos" (laptops open on the sample instead of a webcam).
2. **Camera.** Full screen. Today's sun path drawn live over the real sky from the phone's sensors, a bearing scale low in
   the frame that turns with the phone (ticks every 15°, compass names, gold where the sweep has covered, the sun as a gold
   dot), the sun ringed when it is in frame, a one-line hint, shots taken by themselves when the phone is steady on a new
   slice of the view (the shutter still works), Done. Sheets for a refused location and for errors.
3. **Processing.** The last frame stays on screen; each frame then appears with SAM's skyline traced onto it like a pen and
   the sun ringed where SAM found it; a status pill ("Tracing the skyline · 2 of 6 · sun found").
4. **Result.** The photo is the whole screen. The time large and thin at the top with "In sun" / "In shade"; the sun
   crossing the user's own sky, hidden by the real skyline (the SAM mask), drawn at its true angular size, the scene warm
   in sun and cool in shade; the month's path drawn on the sky, solid where lit and dotted where blocked, with hour marks;
   compass bearings ticked along the skyline; the contour kept faintly. Over the bottom: hours of sun a day by month
   (tap a month), the answer in one sentence ("Afternoon sun, 1 to 4 hours a day. From about 13:30 until about 16:45.
   Most in March, least in June."), and a day band from sunrise to sunset, gold where lit, with a cursor to drag. On
   arrival the skyline traces itself, the sun runs its day, then the words come up. "Details" (or a swipe up) opens a sheet:
   checked against the real sun (a crop of the real sun with our ring and the compass error), how much of the horizon was
   covered, good for / not for / tip, a question box answered by Spark, Share (a 4:5 card), Another spot.

## Design (decisions of 3 November)

Four rounds, each judged by Ayman: a light "startup" page (generic), a dusk page with a serif and a sun dial (still a
template), a dark card list with a system (grammar without a voice), then the one that stuck: **the photo is the
screen and the interface recedes.** References accepted: Apple Camera in iOS 26 for the viewfinder, Flighty for the
result's density and calm, Apple Weather for the sun strip; Halide rejected as outdated. Rules that follow: black around
the photo, one gold (the sun and what the user has done), the phone's own type, drawn icons only, frosted pills, a 4-point
grid and a five-step type scale, and everything drawn on the photo must be measured data (contour, path, bearings), never
decoration. Removed on purpose: the spot-type question, the sun dial, the bars, the video-style scrubber, any sample made
from Ayman's own balcony.

## Status

- Verified: geometry on synthetic data (horizon to 0.02° median, heading and tilt from the sun exact, lens from the sun
  with known tilt exact); the full chain on a real balcony frame and on the generated sample; the page end to end on the
  Mac at phone size and, for the capture flow, from an iPhone indoors (sensors, 4K stream, upload, SAM, Spark). The live service on Cloud Run ran the same sample upload end to end (SAM, the pose from the sun, the hours,
  Spark's verdict and a question, every asset) in 33 s on 3 November.
- Not yet verified: the phone's compass against the sun (the balcony sweep with the sun in a shot), the lens default for
  the camera stream, the monthly hours against a time-lapse, the bearing scale and the processing sequence on the phone
  itself, the share sheet on iOS.
- Not yet built: the public repository, the video, the submission.
