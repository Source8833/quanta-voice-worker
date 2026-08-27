"""
Quanta Voice — RunPod serverless handler.
QuantaCore Labs.

Quanta Voice is Kokoro-82M. This is not a new choice: `af_heart` was already
Quanta's voice in the Suite (quanta_tools/voice_engine.py), described there as
"Warm, empathic female — Quanta default". This worker moves that same engine,
the same model files and the same default onto RunPod so the hosted app can
speak in Quanta's own voice instead of the browser's robot.

WHY CPU AND NOT A GPU
Kokoro is 82M parameters and `kokoro-onnx` is built for CPU. A 24 GB card
would cost about five times as much per hour and sit idle. On serverless CPU
this runs at roughly $0.14/hr while actually speaking and $0 at rest.

WHAT THIS RETURNS
A base64 WAV, 24 kHz, PCM 16-bit — the same format the Suite writes to disk.
The gateway (`functions/api/tts.js`) streams it on to the browser.
"""

import base64
import io
import os

import runpod
import soundfile as sf
from kokoro_onnx import Kokoro

MODEL_PATH = os.environ.get("KOKORO_MODEL", "/models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("KOKORO_VOICES", "/models/voices-v1.0.bin")

DEFAULT_VOICE = os.environ.get("QUANTA_VOICE", "af_heart")

# The gateway already truncates at a sentence boundary; this is the backstop
# for anything reaching the endpoint directly. Synthesis time scales with
# length, and an unbounded request is billable compute.
MAX_CHARS = int(os.environ.get("QUANTA_TTS_MAX_CHARS", "1200"))

# Kokoro's published voices, exactly as the Suite lists them. An unknown name
# falls back to Quanta's own rather than erroring — a wrong voice is a far
# better failure than silence.
VOICES = {
    "af_heart", "af_bella", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_onyx", "am_puck",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
}

# Loaded at import, on purpose: this runs during worker boot, so the model is
# resident before the first job is dispatched rather than being paid for by
# whoever happens to speak first.
KOKORO = Kokoro(MODEL_PATH, VOICES_PATH)


def _clamp(value, low, high, fallback):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return fallback


def handler(job):
    payload = job.get("input") or {}

    text = str(payload.get("text") or "").strip()
    if not text:
        return {"error": "No text to speak."}
    text = text[:MAX_CHARS]

    voice = payload.get("voice") or DEFAULT_VOICE
    if voice not in VOICES:
        voice = DEFAULT_VOICE

    speed = _clamp(payload.get("speed", 1.0), 0.5, 2.0, 1.0)

    samples, sample_rate = KOKORO.create(
        text=text, voice=voice, speed=speed, lang="en-us"
    )

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    audio = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "audio": audio,
        "format": "wav",
        "sample_rate": sample_rate,
        "voice": voice,
        "characters": len(text),
    }


runpod.serverless.start({"handler": handler})
