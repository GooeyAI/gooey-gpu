from fastapi import FastAPI

import controlnet
import deforum
import diffusion
import gooey_gpu
import lv

app = FastAPI()
app.include_router(diffusion.app)
app.include_router(deforum.app)
app.include_router(lv.app)
app.include_router(controlnet.app)
gooey_gpu.register_app(app)
