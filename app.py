"""
NyzMind Kokoro TTS API — minimal ONNX edition
Uses kokoro-onnx directly with pre-downloaded model files.
Optimized for 512MB RAM free tier.
"""
import os
import gc
import time
import tempfile
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="NyzMind Kokoro TTS", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None
_model_dir = os.path.join(tempfile.gettempdir(), "kokoro-models")

def get_engine():
    """Load the Kokoro ONNX engine lazily."""
    global _engine
    if _engine is not None:
        return _engine

    print("[Kokoro] Loading ONNX engine...")
    t0 = time.time()

    os.makedirs(_model_dir, exist_ok=True)

    # Download model files if not present
    # int8 model = 92MB (smallest, best for 512MB RAM)
    model_path = os.path.join(_model_dir, "kokoro-v1.0.int8.onnx")
    voices_path = os.path.join(_model_dir, "voices-v1.0.bin")

    if not os.path.exists(model_path):
        print("[Kokoro] Downloading int8 model (92MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
            model_path
        )
        print(f"[Kokoro] Model downloaded: {os.path.getsize(model_path) / 1024 / 1024:.1f}MB")

    if not os.path.exists(voices_path):
        print("[Kokoro] Downloading voices (28MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            voices_path
        )
        print(f"[Kokoro] Voices downloaded: {os.path.getsize(voices_path) / 1024 / 1024:.1f}MB")

    from kokoro_onnx import Kokoro
    _engine = Kokoro(model_path, voices_path)

    print(f"[Kokoro] Engine loaded in {time.time() - t0:.1f}s")
    return _engine


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


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field("af_heart")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "kokoro-82m-onnx", "voices": len(VOICES), "loaded": _engine is not None}


@app.get("/voices")
async def voices():
    return {"voices": VOICES, "default": "af_heart"}


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    if req.voice not in VALID_VOICE_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid voice '{req.voice}'")

    try:
        engine = get_engine()

        import numpy as np
        import soundfile as sf
        import io

        # kokoro_onnx returns (sample_rate, audio_array)
        samples, sample_rate = engine.create(req.text, req.voice, speed=req.speed)

        # Convert to WAV
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format='WAV', subtype='PCM_16')
        wav_bytes = buf.getvalue()

        del samples, buf
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
    return {"name": "NyzMind Kokoro TTS", "version": "3.0.0", "engine": "onnx", "endpoints": ["/health", "/voices", "/api/tts"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    # Single worker to minimize memory
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", workers=1)
