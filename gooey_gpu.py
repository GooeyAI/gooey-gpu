import contextlib
import gc
import io
import math
import os
import threading
import traceback
import typing
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from multiprocessing.pool import Pool
from time import time

import PIL.Image
import PIL.ImageOps
import numpy as np
import requests
import scipy
import sentry_sdk
import torch
import transformers
from accelerate import cpu_offload_with_hook
from starlette.responses import JSONResponse

try:
    from diffusers import ConfigMixin
except ImportError:
    ConfigMixin = None


DEVICE_ID = os.environ.get("DEVICE_ID", "").strip() or "cuda:0"
REDIS_HOST = os.environ.get("REDIS_HOST", "").strip()
SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "").strip() or "1")
DISABLE_CPU_OFFLOAD = bool(
    int(os.environ.get("DISABLE_CPU_OFFLOAD", "").strip() or "0")
)


if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        send_default_pii=True,
    )
    print("🛰️ Sentry error tracking enabled.")

if REDIS_HOST:
    import redis
    from redis.exceptions import LockError
    from redis.lock import Lock

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

    redis_client = redis.Redis(REDIS_HOST)
    gpu_lock = GpuLock(redis_client, f"gpu-locks/{DEVICE_ID}")
else:
    gpu_lock = threading.RLock()


def endpoint(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        s = time()
        print(f"---> {fn.__name__}: {args!r} {kwargs!r}")
        try:
            response = fn(*args, **kwargs)
        except GpuFuncException as e:
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


M = typing.TypeVar("M", bound=torch.nn.Module | ConfigMixin | transformers.Pipeline)


@contextlib.contextmanager
def use_models(*models: M):
    # register models for offloading
    for obj in models:
        register_cpu_offload(obj)
    try:
        # run the code that uses the model.
        # this should automatically move it to gpu when model's forward() is called.
        yield
    finally:
        if DISABLE_CPU_OFFLOAD:
            return
        # offload to cpu
        for obj in models:
            register_cpu_offload(obj, offload=True)
        # free memory
        gc.collect()
        torch.cuda.empty_cache()


def register_cpu_offload(obj: M, offload: bool = False):
    # pytorch models
    if isinstance(obj, torch.nn.Module):
        module_register_cpu_offload(obj, offload)
    # transformers
    elif isinstance(obj, transformers.Pipeline):
        module_register_cpu_offload(obj.model, offload)
    # diffusers
    elif isinstance(obj, ConfigMixin):
        module_names, _, _ = obj.extract_init_dict(dict(obj.config))
        for name in module_names.keys():
            module = getattr(obj, name)
            if isinstance(module, torch.nn.Module):
                module_register_cpu_offload(module, offload)
    else:
        raise ValueError(f"Not sure how to offload a `{type(obj)}`")


# _saved_hooks = {}


def module_register_cpu_offload(module: torch.nn.Module, offload: bool = False):
    if offload:
        module.to("cpu")
    else:
        module.to(DEVICE_ID)
    # try:
    #     hook = _saved_hooks[module]
    # except KeyError:
    #     module.to("cpu")
    #     _, hook = cpu_offload_with_hook(module, DEVICE_ID)
    #     _saved_hooks[module] = hook
    # if offload:
    #     hook.offload()


P = typing.ParamSpec("P")
R = typing.TypeVar("R")


class Shared:
    task_registry: dict[str, typing.Callable] = {}
    gpu_pool: Pool


def gpu_task(fn: typing.Callable[P, R]) -> typing.Callable[P, R]:
    task_id = f"{fn.__module__}.{fn.__qualname__}"
    Shared.task_registry[task_id] = fn
    print(f"✅ [{os.getpid()}] [{task_id}]")

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return Shared.gpu_pool.apply(gpu_worker, [task_id, time(), args, kwargs])

    return wrapper


def gpu_worker(task_id, s, args, kwargs):
    fn = Shared.task_registry[task_id]
    print(f"⚡️ [{os.getpid()}] [{task_id}] acquired {DEVICE_ID} in {time() - s:.3f}s")
    s = time()
    try:
        # run the func
        return fn(*args, **kwargs)
    except Exception as e:
        # avoids piping exception stacktraces, which might cause memory leaks - https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
        raise GpuFuncException(_response_for_exc(e))
    finally:
        # free memory
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"⚡️ [[{os.getpid()}] {task_id}]️ released {DEVICE_ID} in {time() - s:.3f}s"
        )


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


class GpuFuncException(Exception):
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
        im_pil = downscale_img(im_pil, max_size)
    return im_pil


def downscale_img(im_pil: PIL.Image.Image, max_size: (int, int)) -> PIL.Image.Image:
    downscale_factor = get_downscale_factor(im_size=im_pil.size, max_size=max_size)
    if downscale_factor:
        im_pil = PIL.ImageOps.scale(im_pil, downscale_factor)
        print(f"Resize image by {downscale_factor:.3f}x = {im_pil.size}")
    return im_pil


def get_downscale_factor(*, im_size: (int, int), max_size: (int, int)) -> float | None:
    downscale_factor = math.sqrt(
        (max_size[0] * max_size[1]) / (im_size[0] * im_size[1])
    )
    if downscale_factor < 0.99:
        return downscale_factor
    else:
        return None


def upload_images(images: list[PIL.Image.Image], upload_urls: list[str]):
    apply_parallel(upload_image, images, upload_urls)


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


def apply_parallel(fn, *iterables):
    threads = [threading.Thread(target=fn, args=args) for args in zip(*iterables)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def map_parallel(fn, it):
    with ThreadPoolExecutor(max_workers=len(it)) as pool:
        return list(pool.map(fn, it))


def upload_audio(audio: np.ndarray, url: str, rate: int = 16_000):
    # The resulting audio output can be saved as a .wav file:
    f = io.BytesIO()
    scipy.io.wavfile.write(f, rate=rate, data=audio)
    audio_bytes = f.getvalue()
    # upload to given url
    r = requests.put(url, headers={"Content-Type": "audio/wav"}, data=audio_bytes)
    r.raise_for_status()
