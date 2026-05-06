#!/usr/bin/env python3
"""
serve.py
DEX inference server.

  python3 serve.py

Clients connect via WebSocket. The server:
  1. Receives 20Hz frames from the browser relay
     {"t": ms, "smooth_uS": float, "delta": float, "velocity": float}
  2. Runs a sliding window detector
  3. When a candidate event is detected, classifies it and emits:
     {"type": "call", "label": "read"|"other", "confidence": float, "event_id": str}
  4. Receives feedback:
     {"type": "feedback", "event_id": str, "label": bool}
     and appends to feedback.jsonl

Endpoints:
  ws://host/stream   — main bidirectional channel (browser ↔ DEX)
  GET /health        — {"status": "ok", "model": meta}
"""

import asyncio
import collections
import json
import pathlib
import time
import uuid

import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = pathlib.Path("dex.joblib")
FEEDBACK_LOG  = pathlib.Path("feedback.jsonl")
FEATURES      = ["amplitude", "attack_s", "release_s", "baseline_uS"]

# Sliding window — how many frames to look back when an event is detected
WINDOW_FRAMES = 60        # 3 seconds at 20Hz

# Detector thresholds — tuned conservatively to reduce false positives
# A candidate fires when delta drops by at least this much within the window
DELTA_DROP_THRESH   = 11 / 8  # normalised delta units (firmware scale)
MIN_DURATION_FRAMES = 8     # ~400ms minimum event length
COOLDOWN_FRAMES     = 40    # ~2s between candidates

# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    if not MODEL_PATH.exists():
        print(f"[dex] {MODEL_PATH} not found — run train.py first")
        return None, None
    bundle = joblib.load(MODEL_PATH)
    print(f"[dex] model loaded — meta: {bundle['meta']}")
    return bundle["model"], bundle["meta"]

model, model_meta = load_model()

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="DEX")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": model_meta}

# ── Per-connection session state ──────────────────────────────────────────────

class SessionState:
    def __init__(self):
        self.window   = collections.deque(maxlen=WINDOW_FRAMES)
        self.cooldown = 0          # frames remaining before next candidate allowed
        self.pending  = {}         # event_id → candidate features (awaiting feedback)

    def push(self, frame: dict):
        """Ingest one frame. Returns a call dict if an event is detected, else None."""
        uS      = frame.get("smooth_uS")
        delta   = frame.get("delta")

        if uS is None or delta is None:
            return None

        self.window.append({"uS": uS, "delta": delta, "t": frame.get("t", 0)})

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if len(self.window) < MIN_DURATION_FRAMES + 4:
            return None

        return self._detect()

    def _detect(self):
        frames = list(self.window)
        deltas  = [f["delta"] for f in frames]
        uS_vals = [f["uS"]    for f in frames]

        # Use a 5-second window (100 frames at 20Hz)
        recent = deltas[-100:]
        uS_recent = uS_vals[-100:]

        if len(recent) < 20:  # need at least 1 second of data
            return None

        # Smooth the recent deltas to reduce spike sensitivity
        smoothed_recent = np.convolve(recent, np.ones(5)/5, mode='valid')
        peak_idx_r = np.argmax(smoothed_recent) + 2  # offset for convolution
        peak = recent[peak_idx_r]

        # ── Hard reject heuristics ────────────────────────────────────────────
        if max(uS_recent) > 14.0:          return None  # rail artifact
        if peak_idx_r < 5:                 return None  # peak too early (not enough pre-peak data)

        # Check if the signal was rising into the peak
        pre_peak = recent[:peak_idx_r]
        if len(pre_peak) < 2 or (pre_peak[-1] - pre_peak[0]) <= 0:
            return None  # not rising into the peak

        rise_s = peak_idx_r * 0.05
        if rise_s < 0.2:                   return None  # too fast (mechanical artifact)

        # Check post-peak
        post_peak = recent[peak_idx_r:]
        if len(post_peak) < MIN_DURATION_FRAMES:
            return None

        trough_idx_r = np.argmin(post_peak)
        trough = post_peak[trough_idx_r]
        duration_frames = trough_idx_r

        if duration_frames < MIN_DURATION_FRAMES:
            return None

        # Relative amplitude check (trough must be at least 30% below peak)
        if (peak - trough) / peak < 0.3:
            return None

        # Measure attack and release
        half_level = peak - (peak - trough) * 0.5
        half_frames = None
        for i, d in enumerate(post_peak):
            if d <= half_level:
                prev_d = post_peak[i-1] if i > 0 else peak
                half_frames = i - 1 + (half_level - d) / (prev_d - d)
                break
        if half_frames is None:
            half_frames = duration_frames * 0.5
        release_frames = duration_frames - half_frames

        # Robust baseline (1 second before peak)
        pre_peak_start = max(0, peak_idx_r - 20)
        pre_peak_uS = uS_recent[pre_peak_start:peak_idx_r]
        baseline_uS = float(np.mean(pre_peak_uS)) if pre_peak_uS else uS_recent[0]

        # Amplitude in delta units
        amplitude = peak - trough

        features = {
            "amplitude":   round(abs(amplitude), 4),
            "attack_s":    round(half_frames * 0.05, 3),
            "release_s":   round(release_frames * 0.05, 3),
            "baseline_uS": round(baseline_uS, 4),
        }

        self.cooldown = COOLDOWN_FRAMES
        return features

    def classify(self, features: dict) -> dict:
        if model is None:
            # No model yet — return candidate for labeling only
            return {"label": "unknown", "confidence": 0.0}

        X = np.array([[features[f] for f in FEATURES]])
        prob  = model.predict_proba(X)[0]
        pred  = int(model.predict(X)[0])
        label = "read" if pred == 1 else "other"
        conf  = float(prob[pred])
        return {"label": label, "confidence": round(conf, 3)}


# ── WebSocket handler ─────────────────────────────────────────────────────────

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[dex] client connected")
    state = SessionState()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # ── feedback from UI ──
            if msg.get("type") == "feedback":
                event_id = msg.get("event_id")
                fb_label = msg.get("label")   # true / false
                if event_id and event_id in state.pending:
                    record = {
                        "event_id":   event_id,
                        "label":      fb_label,
                        "features":   state.pending.pop(event_id),
                        "ts":         time.time(),
                    }
                    with open(FEEDBACK_LOG, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"[dex] feedback saved: {event_id} → {fb_label}")
                continue

            # ── EDA frame from meter ──
            candidate_features = state.push(msg)
            if candidate_features is None:
                continue

            result     = state.classify(candidate_features)
            event_id   = str(uuid.uuid4())[:8]
            state.pending[event_id] = candidate_features

            call = {
                "type":       "call",
                "event_id":   event_id,
                "label":      result["label"],
                "confidence": result["confidence"],
                "features":   candidate_features,
            }
            await ws.send_text(json.dumps(call))
            print(f"[dex] → {result['label']} ({result['confidence']:.2f})  {event_id}")

    except WebSocketDisconnect:
        print("[dex] client disconnected")
