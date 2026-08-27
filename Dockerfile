# Quanta Voice — Kokoro-82M on RunPod serverless (CPU).
# QuantaCore Labs.
#
# CPU base on purpose: Kokoro is 82M parameters and kokoro-onnx targets CPU.
# A GPU image would be larger, slower to cold start (CUDA init), and about
# five times the hourly rate for no benefit at this size.
FROM python:3.11-slim

# espeak-ng: kokoro-onnx phonemises through it, and without it synthesis fails
#   at runtime rather than at build — the worst place to find out.
# libsndfile1: soundfile writes the WAV through it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The weights are BAKED IN rather than pulled at boot or mounted from a network
# volume. They are only ~336 MB, and baking them means: no volume to pay for
# monthly, no per-cold-start download, and a worker that cannot half-start
# because a download failed. These are the exact files the Suite's
# voice_engine.py fetches, from the same release.
RUN mkdir -p /models \
 && curl -fsSL -o /models/kokoro-v1.0.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
 && curl -fsSL -o /models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Fail the BUILD, not a user's request, if the download was truncated or the
# release moved. A silently empty model file would look like a working image
# and break only when someone asked Quanta to speak.
RUN python -c "import os,sys; \
m=os.path.getsize('/models/kokoro-v1.0.onnx'); v=os.path.getsize('/models/voices-v1.0.bin'); \
print(f'kokoro-v1.0.onnx {m} bytes, voices-v1.0.bin {v} bytes'); \
sys.exit('model file looks wrong' if m < 200_000_000 or v < 10_000_000 else 0)"

COPY handler.py .

ENV KOKORO_MODEL=/models/kokoro-v1.0.onnx \
    KOKORO_VOICES=/models/voices-v1.0.bin \
    QUANTA_VOICE=af_heart

CMD ["python", "-u", "handler.py"]
