import contextlib
import gc
import io
import multiprocessing
import os
import threading
import traceback
import typing
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from time import time

import PIL.Image
import PIL.ImageOps
import redis
import requests
import sentry_sdk
import torch
from fastapi import FastAPI
from redis.exceptions import LockError
from redis.lock import Lock
from starlette.responses import JSONResponse

DEVICE_ID = os.environ.get("DEVICE_ID", "cuda:0")
REDIS_HOST = os.environ.get("REDIS_HOST", "").strip()
SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))


if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        send_default_pii=True,
    )
    print("Sentry error tracking enabled.")


def register_app(app: FastAPI):
    if MAX_WORKERS <= 1:
        return

    @app.on_event("startup")
    def on_startup():
        multiprocessing.set_start_method("spawn")
        app.state.pool = multiprocessing.Pool(MAX_WORKERS)

    @app.on_event("shutdown")
    def on_shutdown():
        app.state.pool.terminate()


def endpoint(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        s = time()
        print(f"---> {fn.__name__}: {args!r} {kwargs!r}")
        try:
            response = fn(*args, **kwargs)
        except GpuException as e:
            return e.response
        except Exception as e:
            return _response_for_exc(e)
        finally:
            # just for good measure - https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Total Time: {time() - s:.3f}s")
        return JSONResponse(response)

    return wrapper


class GpuLock(Lock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.local._count = 0

    def acquire(self, *args, **kwargs) -> bool:
        if self.owned():
            self.local._count += 1
            return True
        s = time()
        rc = super().acquire(*args, **kwargs)
        if rc:
            self.local._count = 1
            print(f"GPU Acquired: {time() - s:.3f}s")
        return rc

    def release(self):
        if not self.owned():
            raise LockError("cannot release un-acquired lock")
        self.local._count -= 1
        if not self.local._count:
            super().release()


if REDIS_HOST:
    redis_client = redis.Redis(REDIS_HOST)
    gpu_lock = GpuLock(redis_client, f"gpu-locks/{DEVICE_ID}")
else:
    gpu_lock = threading.RLock()


@contextlib.contextmanager
def use_models(*models):
    # move to gpu
    for model in models:
        model.to(DEVICE_ID)
    try:
        # run context manager
        yield
    finally:
        # move to cpu
        for model in models:
            model.to("cpu")
        # free memory
        gc.collect()
        torch.cuda.empty_cache()


def run_in_gpu(*, app: FastAPI, fn, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    s = time()
    print(f"⚡️ [{os.getpid()}] acquired {DEVICE_ID}")
    try:
        if MAX_WORKERS > 1:
            return app.state.pool.apply(fn, args, kwargs)
        else:
            with gpu_lock:
                return fn(*args, **kwargs)
    except Exception as e:
        raise GpuException(_response_for_exc(e))
    finally:
        # free memory
        gc.collect()
        torch.cuda.empty_cache()
        print(f"⚡️ [{os.getpid()}] released {DEVICE_ID} in {time() - s:.3f}s")


def _response_for_exc(e):
    traceback.print_exc()
    sentry_sdk.capture_exception(e)
    return JSONResponse(
        {
            "type": type(e).__name__,
            "str": str(e)[:5000],
            "repr": repr(e)[:5000],
            "format_exc": traceback.format_exc()[:5000],
        },
        status_code=500,
    )


class GpuException(Exception):
    def __init__(self, response):
        self.response = response


def download_images(
    urls: typing.List[str],
    max_size: (int, int) = None,
    mode: str = "RGB",
) -> typing.List[PIL.Image.Image]:
    def fn(url):
        return download_image(url, max_size, mode)

    return list(map_parallel(fn, urls))


def download_image(
    url: str,
    max_size: (int, int) = None,
    mode: str = "RGB",
) -> PIL.Image.Image:
    im_bytes = requests.get(url).content
    f = io.BytesIO(im_bytes)
    im_pil = PIL.Image.open(f).convert(mode)
    if max_size:
        im_pil = resize_img_scale(im_pil, max_size)
    return im_pil


def resize_img_scale(im_pil: PIL.Image.Image, max_size: (int, int)) -> PIL.Image.Image:
    factor = (max_size[0] * max_size[1]) / (im_pil.size[0] * im_pil.size[1])
    if 1 - factor > 1e-2:
        im_pil = PIL.ImageOps.scale(im_pil, factor)
        print(f"Resize image by {factor:.3f}x = {im_pil.size}")
    return im_pil


def upload_images(images: list[PIL.Image.Image], upload_urls: list[str]):
    def fn(args):
        return upload_image(*args)

    list(map_parallel(fn, list(zip(images, upload_urls))))


def upload_image(im_pil: PIL.Image.Image, url: str):
    f = io.BytesIO()
    im_pil.save(f, format="PNG")
    im_bytes = f.getvalue()

    r = requests.put(
        url,
        headers={"Content-Type": "image/png"},
        data=im_bytes,
    )
    r.raise_for_status()


def map_parallel(fn, it):
    with ThreadPoolExecutor(max_workers=len(it)) as pool:
        return list(pool.map(fn, it))
