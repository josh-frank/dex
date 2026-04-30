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

        recent = deltas[-40:]  # last 2 seconds

        # Find the true peak, then the trough strictly after it
        peak_idx_r = max(range(len(recent)), key=lambda i: recent[i])
        peak        = recent[peak_idx_r]

        # ── hard reject heuristics ────────────────────────────────────────────
        if max(uS_vals) > 14.0:          return None  # rail artifact
        if peak - recent[0] < 0:         return None  # falling not rising
        rise_s = peak_idx_r * 0.05
        if rise_s < 0.1:                 return None  # mechanical artifact
        # ─────────────────────────────────────────────────────────────────────

        post_peak = recent[peak_idx_r:]
        if len(post_peak) < MIN_DURATION_FRAMES:
            return None

        trough_idx_r = min(range(len(post_peak)), key=lambda i: post_peak[i])
        trough        = post_peak[trough_idx_r]

        if peak - trough < DELTA_DROP_THRESH:
            return None

        duration_frames = trough_idx_r
        if duration_frames < MIN_DURATION_FRAMES:
            return None

        # Measure attack (peak → half-decay) and release (half-decay → trough)
        half_level   = peak - (peak - trough) * 0.5
        half_frames  = next(
            (i for i, d in enumerate(post_peak) if d <= half_level),
            duration_frames // 2
        )
        release_frames = duration_frames - half_frames

        # Baseline: tonic level BEFORE the peak, not during the event
        pre_peak_uS = [f["uS"] for f in frames[:-(40 - peak_idx_r)]] if peak_idx_r > 0 else uS_vals[:3]
        baseline_uS = float(np.mean(pre_peak_uS)) if pre_peak_uS else uS_vals[0]

        # Amplitude in delta units (consistent, real)
        amplitude = peak - trough

        features = {
            "amplitude":   round(abs(amplitude), 4),
            "attack_s":    round(half_frames * 0.05, 3),     # peak → half-decay
            "release_s":   round(release_frames * 0.05, 3),  # real seconds
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
