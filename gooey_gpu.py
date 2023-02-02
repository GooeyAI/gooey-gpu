import contextlib
import gc
import io
import os
import traceback
import typing
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, partial
from time import time

import PIL.Image
import PIL.ImageOps
import redis
import requests
import sentry_sdk
import torch
from starlette.responses import Response

DEVICE_ID = os.environ.get("DEVICE_ID", "cuda:0")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
SENTRY_DSN = os.environ.get("SENTRY_DSN")


if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        send_default_pii=True,
    )


def endpoint(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        s = time()

        print("-*-*-")
        print(f" ---> {fn.__name__}: {args!r} {kwargs!r}")

        try:
            response = fn(*args, **kwargs)

        except Exception as e:
            traceback.print_exc()
            sentry_sdk.capture_exception(e)
            return Response(repr(e), status_code=500)

        finally:
            # just for good measure - https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Total Time: {time() - s:.3f}s")
            print("-*-*-")

        return Response(response)

    return wrapper


def gpu_lock():
    return redis_client.lock(f"gpu-locks/{DEVICE_ID}")


redis_client = redis.Redis(REDIS_HOST)


@contextlib.contextmanager
def use_gpu(*models):
    # move to gpu
    for model in models:
        model.to(DEVICE_ID)
    try:
        s = time()
        with torch.inference_mode():
            yield
        print(f"GPU Time: {time() - s:.3f}s")
    finally:
        # move to cpu
        for model in models:
            model.to("cpu")
        # free memory
        gc.collect()
        torch.cuda.empty_cache()


def download_images(
    urls: typing.List[str],
    max_size: (int, int) = None,
) -> typing.List[PIL.Image.Image]:
    fn = partial(download_image, max_size=max_size)
    with ThreadPoolExecutor(len(urls)) as pool:
        return list(pool.map(fn, urls))


def download_image(url: str, max_size: (int, int) = None) -> PIL.Image.Image:
    im_bytes = requests.get(url).content
    f = io.BytesIO(im_bytes)
    im_pil = PIL.Image.open(f).convert("RGB")
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
    for im_pil, upload_url in zip(images, upload_urls):
        f = io.BytesIO()
        im_pil.save(f, format="PNG")
        im_bytes = f.getvalue()

        r = requests.put(
            upload_url,
            headers={"Content-Type": "image/png"},
            data=im_bytes,
        )
        r.raise_for_status()
