import multiprocessing

from fastapi import FastAPI

import controlnet
import deforum
import diffusion
import gooey_gpu
import lv
import nvidia_nemo

app = FastAPI()


app.include_router(diffusion.app)
app.include_router(deforum.app)
app.include_router(lv.app)
app.include_router(controlnet.app)
app.include_router(nvidia_nemo.app)


@app.on_event("startup")
def on_startup():
    # https://github.com/pytorch/pytorch/issues/40403
    gooey_gpu.Shared.gpu_pool = multiprocessing.get_context("spawn").Pool(
        gooey_gpu.MAX_WORKERS
    )
    # start worker processes
    gooey_gpu.Shared.gpu_pool.map_async(_dummy, range(gooey_gpu.MAX_WORKERS))


@app.on_event("shutdown")
def on_shutdown():
    # terminate worker processes
    gooey_gpu.Shared.gpu_pool.terminate()


def _dummy(*_):
    pass
