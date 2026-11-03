"""Ask Muse Spark what is in a recovered environment map. Untested until META_API_KEY is available."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

API_URL = "https://api.meta.ai/v1/chat/completions"
DEFAULT_MODEL = "muse-spark-1.3"

PROMPT = (
    "This image is a low-resolution panoramic reconstruction of a room, recovered from the reflection on a person's cornea in a "
    "selfie. Black areas are outside the reflection. Left-right is as the person sees the room (they face the camera). "
    "List what you can identify: light sources, windows, screens, people (including whoever holds the camera), furniture and "
    "the room's shape. For each item give its direction (left, centre, right; up, level, down) and your confidence from 0 to 1. "
    "Then say in one sentence what a stranger could learn about where this person is. Do not guess beyond the evidence."
)


def load_key() -> str | None:
    """META_API_KEY from the environment, else from a git-ignored .env file next to the project (KEY=value lines)."""
    key = os.environ.get("META_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("META_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def describe(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: int = 180,
    max_tokens: int = 4000,
    reasoning_effort: str | None = "low",
    prompt: str = PROMPT,
) -> str:
    """Spark reasons before it answers and the reasoning counts against max_tokens (a 700-token budget came back empty, all
    of it reasoning), so the budget is generous and the effort low by default."""
    key = api_key or load_key()
    if not key:
        raise SystemExit("META_API_KEY is not set (environment variable, or META_API_KEY=... in catchlight/.env).")
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    if r.status_code == 400 and reasoning_effort:
        payload.pop("reasoning_effort")
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not content:
        usage = data.get("usage", {})
        raise SystemExit(
            f"Empty answer: finish_reason={choice.get('finish_reason')}, usage={usage}. Raise max_tokens or lower reasoning_effort."
        )
    return content
