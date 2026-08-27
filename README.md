# Quanta Voice — RunPod serverless worker

Text to speech for Quanta, by **QuantaCore Labs**.

Quanta Voice is **Kokoro-82M**, run through `kokoro-onnx`. The default voice is
`af_heart` — the same one already chosen in the Quanta Suite, described there
as *"Warm, empathic female — Quanta default"*. This worker moves that engine,
those exact model files and that default onto RunPod so the hosted app speaks
in Quanta's own voice rather than the browser's built-in synthesiser.

## Why CPU

Kokoro is 82M parameters and `kokoro-onnx` targets CPU. A 24 GB GPU would cost
roughly five times as much per hour, cold-start slower (CUDA init), and sit
idle. On serverless CPU this is about **$0.14/hr while actually speaking and
$0 at rest**.

## Request

```json
{ "input": { "text": "Love for the world. Creating the future.", "voice": "af_heart", "speed": 1.0 } }
```

| Field | Default | Notes |
|---|---|---|
| `text` | required | Truncated at `QUANTA_TTS_MAX_CHARS` (1200) |
| `voice` | `af_heart` | Unknown names fall back to the default, never error |
| `speed` | `1.0` | Clamped to 0.5–2.0 |

Voices: `af_heart`, `af_bella`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`,
`am_onyx`, `am_puck`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`.

## Response

```json
{ "audio": "<base64 WAV>", "format": "wav", "sample_rate": 24000, "voice": "af_heart", "characters": 39 }
```

24 kHz, PCM 16-bit — the same format the Suite writes to disk.

## Design notes

- **The weights are baked into the image**, not pulled at boot or mounted from
  a network volume. They are only ~336 MB, so this costs no monthly storage, no
  per-cold-start download, and removes a way for a worker to half-start.
- **The build asserts the model file sizes.** A truncated download would
  otherwise produce an image that looks fine and fails only when someone asks
  Quanta to speak.
- **The model loads at import**, during worker boot, so it is resident before
  the first job is dispatched rather than being paid for by whoever speaks
  first.
- **`espeak-ng` is installed at the system level.** `kokoro-onnx` phonemises
  through it; without it synthesis fails at runtime rather than at build.

## Attribution

Kokoro-82M is by hexgrad. `kokoro-onnx` is by thewh1teagle. Model files are
fetched from the `model-files-v1.0` release of `thewh1teagle/kokoro-onnx`.
Check both licences before shipping this publicly and record them on the
licensing page alongside the other model foundations.
