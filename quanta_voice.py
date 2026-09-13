"""
Quanta's voice, on the web — the same voice QOS speaks.

The source of truth is `Qos/shell/src/voice.rs`; this is its port to the
worker, so iamquanta.ca and Quanta OS sound like one person:

  THE VOICE    65% af_bella + 35% af_heart at 0.99 speed — measured closest to
               the founder's reference clip (2026-09-13) and chosen by ear.
  THE CADENCE  spoken phrase by phrase, with the pauses, the pace per phrase
               and the softened sentence ends tuned against that clip, and a
               calmer row of the style table (lift 0.7).
  THE TONE     steady (the design), warm (someone is struggling), bright
               (hello, thanks, good news) or clear (numbers, status, steps):
               each a change of pauses, pace, softness, style row and a tilt of
               the blend. The browser decides which (app.js, VOICE_TONE).

The numbers here must match voice.rs. They are small, so they are copied
rather than shared, and scripts/test-gateway.mjs checks the two agree.
"""

import re

import numpy as np

SAMPLE_RATE = 24000
BLEND = {"af_bella": 0.65, "af_heart": 0.35}
SPEED = 0.99
CADENCE = {"comma": 0.62, "sentence": 1.5, "paragraph": 1.9, "first": 0.95, "middle": 0.95, "last": 0.95, "fade": 0.62, "style": 0.7}
TONES = ("steady", "warm", "bright", "clear")


def tone_cadence(tone):
    c = dict(CADENCE)
    if tone == "warm":
        c.update(comma=c["comma"] * 1.3, sentence=c["sentence"] * 1.25, paragraph=c["paragraph"] * 1.2,
                 first=c["first"] * 0.93, middle=c["middle"] * 0.93, last=c["last"] * 0.88,
                 fade=max(c["fade"] * 0.8, 0.3), style=min(c["style"] * 1.25, 3.0))
    elif tone == "bright":
        c.update(comma=c["comma"] * 0.65, sentence=c["sentence"] * 0.62, paragraph=c["paragraph"] * 0.7,
                 first=c["first"] * 1.05, middle=c["middle"] * 1.08, last=c["last"] * 1.02,
                 fade=min(c["fade"] * 1.35, 1.0), style=c["style"] * 0.75)
    elif tone == "clear":
        c.update(comma=c["comma"] * 0.55, sentence=c["sentence"] * 0.6, paragraph=c["paragraph"] * 0.7,
                 first=c["first"] * 1.03, middle=c["middle"] * 1.05,
                 fade=min(c["fade"] * 1.45, 1.0), style=min(c["style"] * 1.1, 3.0))
    return c


def tone_blend(tone):
    tilt = {("warm", "af_heart"): 1.7, ("bright", "af_bella"): 1.35}
    return {name: w * tilt.get((tone, name), 1.0) for name, w in BLEND.items()}


def speakable(text):
    """Markdown read the way a person reads it: marks gone, code not read out."""
    out, in_code, said_code = [], False, False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            if in_code and not said_code:
                out.append("The code is on screen.")
                said_code = True
            continue
        if in_code or not line:
            continue
        line = re.sub(r"^[#>]+\s*", "", line)
        line = re.sub(r"^(?:[-*•]|\d{1,2}\.)\s+", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"https?://\S+", "a link", line)
        line = line.replace("**", "").replace("__", "").replace("`", "").replace(" — ", ", ").replace("—", ", ")
        if not re.search(r"[.!?:]$", line):
            line += "."
        out.append(line)
    return " ".join(out)


def plan(text, c):
    """Sentences, then phrases at their commas (one or two words join a
    neighbour), each with its speed and softened end, and the silences."""
    pieces = []
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", speakable(text)) if s.strip()]
    for si, sentence in enumerate(sentences):
        phrases = [p for p in re.split(r"(?<=[,;:])\s+", sentence.strip()) if p]
        merged = []
        for p in phrases:
            if merged and len(merged[-1].split()) <= 2:
                merged[-1] += " " + p
            else:
                merged.append(p)
        if len(merged) > 1 and len(merged[-1].split()) <= 2:
            tail = merged.pop()
            merged[-1] += " " + tail
        n = len(merged)
        for i, p in enumerate(merged):
            last = i + 1 == n
            speed = (c["first"] + c["last"]) * 0.5 if n == 1 else c["first"] if i == 0 else c["last"] if last else c["middle"]
            pieces.append(("say", p, speed, c["fade"] if last else c["fade"] ** 0.5))
            if not last:
                pieces.append(("pause", c["comma"]))
        if si + 1 < len(sentences):
            pieces.append(("pause", c["sentence"]))
    return pieces


class QuantaVoice:
    def __init__(self, kokoro):
        self.kokoro = kokoro
        self.tables = {tone: self._table(tone) for tone in TONES}

    def _table(self, tone):
        weights = tone_blend(tone)
        total = sum(weights.values())
        table = sum(np.asarray(self.kokoro.voices[name], dtype=np.float32) * (w / total) for name, w in weights.items())
        # The style row a phrase reads is its length times the lift (voice.rs
        # synth_at); kokoro-onnx reads row len(tokens), so the rows are remapped.
        lift = tone_cadence(tone)["style"]
        rows = table.shape[0]
        idx = np.clip(np.round(np.arange(rows) * lift).astype(int), 1, rows - 1)
        return table[idx].astype(np.float32)

    def speak(self, text, tone="steady", speed=1.0):
        tone = tone if tone in TONES else "steady"
        c = tone_cadence(tone)
        table = self.tables[tone]
        out = []
        for piece in plan(text, c):
            if piece[0] == "pause":
                out.append(np.zeros(int(SAMPLE_RATE * piece[1]), dtype=np.float32))
                continue
            _, phrase, phrase_speed, fade = piece
            audio, _ = self.kokoro.create(text=phrase, voice=table, speed=float(np.clip(SPEED * phrase_speed * speed, 0.5, 2.0)), lang="en-us")
            audio = np.asarray(audio, dtype=np.float32).copy()
            n = len(audio)
            start = int(n * 0.55)
            if n > start + 1:
                t = np.linspace(0.0, 1.0, n - start, dtype=np.float32)
                audio[start:] *= 1.0 - (1.0 - fade) * t * t
            out.append(audio)
        if not out:
            return np.zeros(0, dtype=np.float32)
        # The next sentence arrives as the next request: its pause goes here.
        out.append(np.zeros(int(SAMPLE_RATE * c["sentence"]), dtype=np.float32))
        return np.concatenate(out)
