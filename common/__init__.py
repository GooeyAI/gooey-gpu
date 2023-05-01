from fastapi import APIRouter

from common import audio_ldm, controlnet, diffusion, suno_ai_bark, deepfloyd, whisper

app = APIRouter()
app.include_router(audio_ldm.app)
app.include_router(controlnet.app)
app.include_router(diffusion.app)
app.include_router(suno_ai_bark.app)
app.include_router(whisper.app)
app.include_router(deepfloyd.app)
