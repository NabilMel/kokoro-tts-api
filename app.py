"""
NyzMind Kokoro TTS API
======================
A free, self-hosted Kokoro TTS API running on HuggingFace Spaces.

Provides a simple REST API for text-to-speech using the Kokoro 82M model.
Default voice: af_heart (warm, natural, feminine — perfect for Nyz).

Endpoints:
  GET  /health   — health check
  GET  /voices   — list available voices
  POST /api/tts  — generate speech from text
"""

import io
import time
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Kokoro imports — loaded at startup
from kokoro import KModel, KPipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="NyzMind Kokoro TTS", version="1.0.0")

# CORS — allow requests from NyzMind web app and Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

print("[Kokoro] Loading model...")
t0 = time.time()
model = KModel().eval()
pipeline = KPipeline(model=model, language_code='a')  # 'a' = American English
print(f"[Kokoro] Model loaded in {time.time() - t0:.1f}s")

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
# Request / Response models
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
    return {"status": "ok", "model": "kokoro-82m", "voices": len(VOICES)}

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
        # Generate audio using Kokoro
        # KPipeline returns (graphemes, phonemes, audio) generator
        # We collect all chunks and concatenate
        chunks = []
        for graphemes, phonemes, audio in pipeline(req.text, voice=req.voice, speed=req.speed):
            if audio is not None:
                chunks.append(audio if isinstance(audio, np.ndarray) else np.array(audio))

        if not chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Concatenate all audio chunks
        full_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, full_audio, 24000, format='WAV', subtype='PCM_16')
        wav_bytes = buf.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="tts.wav"',
                "X-Voice": req.voice,
                "X-Duration": f"{len(full_audio) / 24000:.2f}",
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
        "version": "1.0.0",
        "endpoints": ["/health", "/voices", "/api/tts"],
        "default_voice": "af_heart",
    }

# ---------------------------------------------------------------------------
# Main entry point (for HuggingFace Spaces Docker SDK)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
