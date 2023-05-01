import multiprocessing
import os

from fastapi import FastAPI

import gooey_gpu

app = FastAPI()
variant = __import__(os.environ["VARIANT"])
app.include_router(variant.app)


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
