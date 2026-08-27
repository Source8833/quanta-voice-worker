# Quanta Voice — Kokoro-82M on RunPod serverless.
# QuantaCore Labs.
#
# ---------------------------------------------------------------------
# THIS WAS A CPU IMAGE. MEASUREMENT SAID GPU, AND SAID IT ON COST.
#
# The first version targeted CPU on the reasoning that 82M parameters cannot
# need a graphics card and CPU is cheaper per hour. Measured on the live
# endpoint (2 vCPU cpu3g, 2026-08-27):
#
#     cold : 65.7 s of compute for  5.72 s of audio  = 11.5x realtime
#     warm : 42.6 s of compute for  3.24 s of audio  = 13.1x realtime
#
# Forty-two seconds before Quanta says one sentence. Unusable for a voice.
#
# The cost reasoning was also backwards, because you pay for TIME, not for
# how big the card is:
#
#     CPU 2 vCPU @ $0.10/hr at 13.1x realtime  = $0.00036  per second of speech
#     GPU AMPERE_24 @ $0.69/hr at ~0.05x       = $0.0000096 per second of speech
#
# Seven times the hourly rate, roughly a fortieth of the cost per sentence,
# because the work finishes almost immediately. Cheap-per-hour lost to
# cheap-per-job.
# ---------------------------------------------------------------------
# CUDA 12.8, not 12.4: the US pool that actually has stock is Blackwell
# (RTX PRO 6000 MIG), and Blackwell needs 12.8+. The worker's own fitness
# check reports the host driver as CUDA 13.0, which runs a 12.8 runtime fine.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# espeak-ng: kokoro-onnx phonemises through it, and without it synthesis fails
#   at runtime rather than at build — the worst place to find out.
# libsndfile1: soundfile writes the WAV through it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        espeak-ng \
        libsndfile1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# `kokoro-onnx` pulls in `onnxruntime` (CPU) as a dependency. It and
# `onnxruntime-gpu` install the SAME `onnxruntime` module, so the CPU wheel
# silently overwrites the GPU one and the CUDA provider vanishes entirely -
# not disabled, ABSENT from get_available_providers(). Evict it and reinstall
# the GPU build last so it is the copy left on disk.
RUN pip3 install --no-cache-dir -r requirements.txt  && pip3 uninstall -y onnxruntime  && pip3 install --no-cache-dir --force-reinstall "onnxruntime-gpu>=1.22.0"

# FAIL THE BUILD, not a user's request. Without this the image ships happily,
# runs on a GPU worker, quietly uses the CPU, and the only tell is a field in
# the response that somebody has to think to read.
RUN python3 -c "import importlib.metadata as md, sys, onnxruntime as ort; names={d.metadata['Name'].lower() for d in md.distributions()}; print('installed:', sorted(n for n in names if 'onnxruntime' in n)); print('providers:', ort.get_available_providers()); sys.exit('CPU onnxruntime is shadowing the GPU build') if 'onnxruntime' in names else None; sys.exit('onnxruntime-gpu missing') if 'onnxruntime-gpu' not in names else None; sys.exit('no CUDAExecutionProvider compiled in') if 'CUDAExecutionProvider' not in ort.get_available_providers() else None"

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

# Fail the BUILD, not a user's request, if a download was truncated or the
# release moved. A silently empty model file would look like a working image
# and break only when someone asked Quanta to speak.
RUN python3 -c "import os,sys; \
m=os.path.getsize('/models/kokoro-v1.0.onnx'); v=os.path.getsize('/models/voices-v1.0.bin'); \
print(f'kokoro-v1.0.onnx {m} bytes, voices-v1.0.bin {v} bytes'); \
sys.exit('model file looks wrong' if m < 200_000_000 or v < 10_000_000 else 0)"

COPY handler.py .

ENV KOKORO_MODEL=/models/kokoro-v1.0.onnx \
    KOKORO_VOICES=/models/voices-v1.0.bin \
    QUANTA_VOICE=af_heart

CMD ["python3", "-u", "handler.py"]
