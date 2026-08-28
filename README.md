---
title: NyzMind Kokoro TTS
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Neural TTS API for NyzMind — warm, human-like speech
---

# NyzMind Kokoro TTS

A free, self-hosted Kokoro TTS API running on HuggingFace Spaces.

## API

### `POST /api/tts`

**Request:**
```json
{
  "text": "Hello, I am Nyz.",
  "voice": "af_heart",
  "speed": 1.0
}
```

**Response:** Audio file (`audio/wav`)

### `GET /health`

Returns `{"status": "ok"}` if the service is ready.

### `GET /voices`

Returns the list of available voices.

## Voices

Default: `af_heart` (warm, natural, feminine — grade A)

See full list at [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M)
