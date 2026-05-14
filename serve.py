#!/usr/bin/env python3
"""
serve.py
DEX inference server.

  python3 serve.py

Clients connect via WebSocket. The server:
  1. Receives 20Hz frames from the browser relay
     {"t": ms, "smooth_uS": float, "delta": float, "velocity": float}
  2. Runs a two-EMA deviation detector with an armed/fire state machine
  3. When a read is detected, emits:
     {"type": "call", "label": "read", "confidence": float, "event_id": str, "features": {...}}
  4. Receives feedback:
     {"type": "feedback", "event_id": str, "label": bool}
     and appends to feedback.jsonl

Endpoints:
  ws://host/stream   — main bidirectional channel (browser ↔ DEX)
  GET /health        — {"status": "ok", "model": "heuristic"}

── How detection works ───────────────────────────────────────────────────────

Two exponential moving averages track the GSR signal:

  ema_slow  τ ≈ 20s   — tonic baseline (where the signal has been sitting)
  ema_fast  τ ≈ 0.5s  — current level (responds to phasic events quickly)

  deviation = ema_fast - ema_slow

State machine, evaluated every frame:

  IDLE  → ARMED   when deviation crosses RISE_THRESH upward
                  (something is rising above baseline — get ready)

  ARMED → FIRED   when deviation falls back to FALL_THRESH below peak_deviation
                  (the rise has confirmed a fall — it's a read)
                  emits the call with amplitude + duration features

  FIRED → IDLE    after COOLDOWN_FRAMES (prevents re-triggering on the same event)

This is essentially what a human auditor does watching the needle:
  - notice it rising
  - confirm it falls back
  - call it

No training data required. Tunable with three numbers.
"""

import json
import pathlib
import time
import uuid

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

FEEDBACK_LOG = pathlib.Path("feedback.jsonl")

# EMA time constants — expressed as α (per-frame smoothing factor)
# α = 1 - exp(-1 / (τ_seconds * fps))
FPS            = 20
EMA_SLOW_TAU   = 20.0   # seconds — tonic baseline tracker
EMA_FAST_TAU   = 0.5    # seconds — phasic signal tracker
ALPHA_SLOW     = 1 - np.exp(-1 / (EMA_SLOW_TAU * FPS))   # ≈ 0.0049
ALPHA_FAST     = 1 - np.exp(-1 / (EMA_FAST_TAU * FPS))   # ≈ 0.0952

# Detector thresholds
RISE_THRESH    = 0.3    # µS above baseline to arm  (tune up if too noisy)
FALL_RATIO     = 0.4    # must fall this fraction from peak to fire
                        # e.g. 0.4 → peak of 0.5µS needs to drop back to 0.3µS
MIN_ARMED_FRAMES = 4    # ~200ms — ignore sub-blip rises
COOLDOWN_FRAMES  = 60   # ~3s lockout after each call

# Rail artifact rejection — signal above this is sensor saturation
RAIL_US        = 14.0

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
    return {"status": "ok", "model": "heuristic"}

# ── Per-connection session state ──────────────────────────────────────────────

class SessionState:
    def __init__(self):
        self.ema_slow      = None   # initialised on first frame
        self.ema_fast      = None
        self.cooldown      = 0
        self.armed         = False
        self.armed_frames  = 0
        self.peak_dev      = 0.0    # highest deviation seen while armed
        self.peak_uS       = 0.0    # highest smooth_uS seen while armed
        self.arm_time      = 0.0    # timestamp (ms) when armed
        self.pending       = {}     # event_id → features, awaiting feedback

    def push(self, frame: dict):
        """
        Ingest one frame. Returns a features dict if a read fires, else None.
        """
        uS = frame.get("smooth_uS")
        t  = frame.get("t", 0)

        if uS is None:
            return None

        # Hard reject: sensor rail
        if uS > RAIL_US:
            return None

        # Initialise EMAs on first frame
        if self.ema_slow is None:
            self.ema_slow = uS
            self.ema_fast = uS
            return None

        # Update EMAs
        self.ema_slow = self.ema_slow + ALPHA_SLOW * (uS - self.ema_slow)
        self.ema_fast = self.ema_fast + ALPHA_FAST * (uS - self.ema_fast)

        deviation = self.ema_fast - self.ema_slow

        # Cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            # Still track EMAs during cooldown, just don't fire
            return None

        # ── State machine ─────────────────────────────────────────────────────

        if not self.armed:
            # IDLE → ARMED
            if deviation >= RISE_THRESH:
                self.armed        = True
                self.armed_frames = 0
                self.peak_dev     = deviation
                self.peak_uS      = uS
                self.arm_time     = t
        else:
            # ARMED — track peak and wait for fall
            self.armed_frames += 1

            if deviation > self.peak_dev:
                self.peak_dev = deviation
                self.peak_uS  = uS

            fall_threshold = self.peak_dev * (1 - FALL_RATIO)

            if deviation <= fall_threshold:
                # ARMED → FIRED — confirmed rise and fall

                if self.armed_frames < MIN_ARMED_FRAMES:
                    # Too brief — blip, not a read
                    self.armed = False
                    return None

                duration_s  = self.armed_frames / FPS
                amplitude   = self.peak_dev            # peak deviation in µS
                baseline_uS = float(self.ema_slow)     # tonic level at fire time

                features = {
                    "amplitude":   round(amplitude, 4),
                    "duration_s":  round(duration_s, 3),
                    "baseline_uS": round(baseline_uS, 4),
                    "peak_uS":     round(self.peak_uS, 4),
                }

                self.armed    = False
                self.cooldown = COOLDOWN_FRAMES
                return features

            # If deviation drops below zero while armed, abandon — false alarm
            if deviation < 0:
                self.armed = False

        return None

    def confidence(self, features: dict) -> float:
        """
        Heuristic confidence score — no model required.
        Larger amplitude relative to baseline → higher confidence.
        Returns 0.0–1.0.
        """
        amp  = features["amplitude"]
        base = max(features["baseline_uS"], 0.1)  # avoid div/0
        # Sigmoid-ish: 0.5µS deviation = ~0.5 conf, 1.5µS = ~0.88
        ratio = amp / base
        return round(float(1 / (1 + np.exp(-5 * (ratio - 0.15)))), 3)


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

            # ── feedback from UI ──────────────────────────────────────────────
            if msg.get("type") == "feedback":
                event_id = msg.get("event_id")
                fb_label = msg.get("label")   # true / false
                if event_id and event_id in state.pending:
                    record = {
                        "event_id": event_id,
                        "label":    fb_label,
                        "features": state.pending.pop(event_id),
                        "ts":       time.time(),
                    }
                    with open(FEEDBACK_LOG, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"[dex] feedback saved: {event_id} → {fb_label}")
                continue

            # ── EDA frame from meter ──────────────────────────────────────────
            features = state.push(msg)
            if features is None:
                continue

            event_id = str(uuid.uuid4())[:8]
            conf     = state.confidence(features)
            state.pending[event_id] = features

            call = {
                "type":       "call",
                "event_id":   event_id,
                "label":      "read",
                "confidence": conf,
                "features":   features,
            }
            await ws.send_text(json.dumps(call))
            print(f"[dex] READ  amp={features['amplitude']:.3f}µS  "
                  f"dur={features['duration_s']:.2f}s  "
                  f"conf={conf:.2f}  {event_id}")

    except WebSocketDisconnect:
        print("[dex] client disconnected")
