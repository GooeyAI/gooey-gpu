from fastapi import APIRouter

from backports import nvidia_nemo, lv

app = APIRouter()
app.include_router(nvidia_nemo.app)
app.include_router(lv.app)
