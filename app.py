"""
NyzMind Kokoro TTS API — ONNX Runtime edition
=============================================
Uses fastkokoro (ONNX Runtime) instead of PyTorch.
Memory footprint: ~200MB instead of ~1GB.
Optimized for Render.com free tier (512MB RAM).

Endpoints:
  GET  /health   — health check
  GET  /voices   — list available voices
  POST /api/tts  — generate speech from text
"""

import os
import gc
import time
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="NyzMind Kokoro TTS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy model loading — only load when first TTS request comes in
# ---------------------------------------------------------------------------

_engine = None

def get_engine():
    """Load the Kokoro engine lazily (only on first request)."""
    global _engine
    if _engine is not None:
        return _engine

    print("[Kokoro] Loading ONNX engine (first request)...")
    t0 = time.time()

    from fastkokoro import FastKokoro
    _engine = FastKokoro()

    print(f"[Kokoro] Engine loaded in {time.time() - t0:.1f}s")
    return _engine

# ---------------------------------------------------------------------------
# Voice catalog
# ---------------------------------------------------------------------------

VOICES = [
    {"id": "af_heart", "name": "Heart", "traits": "Warm, natural", "gender": "female"},
    {"id": "af_bella", "name": "Bella", "traits": "Fiery, expressive", "gender": "female"},
    {"id": "af_nicole", "name": "Nicole", "traits": "Narration", "gender": "female"},
    {"id": "af_sarah", "name": "Sarah", "traits": "Calm, steady", "gender": "female"},
    {"id": "af_sky", "name": "Sky", "traits": "Bright, airy", "gender": "female"},
    {"id": "af_nova", "name": "Nova", "traits": "Modern, crisp", "gender": "female"},
    {"id": "af_alloy", "name": "Alloy", "traits": "Neutral", "gender": "female"},
    {"id": "af_kore", "name": "Kore", "traits": "Gentle", "gender": "female"},
    {"id": "af_aoede", "name": "Aoede", "traits": "Soft", "gender": "female"},
    {"id": "af_river", "name": "River", "traits": "Flowing", "gender": "female"},
    {"id": "am_michael", "name": "Michael", "traits": "Warm, deep", "gender": "male"},
    {"id": "am_adam", "name": "Adam", "traits": "Neutral", "gender": "male"},
    {"id": "am_eric", "name": "Eric", "traits": "Crisp", "gender": "male"},
    {"id": "am_liam", "name": "Liam", "traits": "Friendly", "gender": "male"},
    {"id": "am_onyx", "name": "Onyx", "traits": "Deep, rich", "gender": "male"},
    {"id": "am_pade", "name": "Pade", "traits": "Calm", "gender": "male"},
]

VALID_VOICE_IDS = {v["id"] for v in VOICES}

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field("af_heart", description="Voice ID (see /voices)")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": "kokoro-82m-onnx", "voices": len(VOICES), "loaded": _engine is not None}

@app.get("/voices")
async def voices():
    return {"voices": VOICES, "default": "af_heart"}

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Generate speech from text. Returns a WAV audio file."""
    if req.voice not in VALID_VOICE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice '{req.voice}'. Valid voices: {sorted(VALID_VOICE_IDS)}"
        )

    try:
        engine = get_engine()

        # Generate audio using fastkokoro
        audio = engine.create(
            req.text,
            voice=req.voice,
            response_format="wav",
        )

        # audio is bytes (WAV format)
        if isinstance(audio, bytes):
            wav_bytes = audio
        elif hasattr(audio, 'read'):
            wav_bytes = audio.read()
        else:
            wav_bytes = bytes(audio)

        gc.collect()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="tts.wav"',
                "X-Voice": req.voice,
                "Cache-Control": "no-cache",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Kokoro] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.get("/")
async def root():
    return {
        "name": "NyzMind Kokoro TTS",
        "version": "2.0.0",
        "engine": "onnx",
        "endpoints": ["/health", "/voices", "/api/tts"],
        "default_voice": "af_heart",
    }

# ---------------------------------------------------------------------------
# Main — bind to PORT env var (Render sets this)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
