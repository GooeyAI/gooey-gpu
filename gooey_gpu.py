import contextlib
import gc
import inspect
import io
import math
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import PIL.Image
import PIL.ImageOps
import requests
import sentry_sdk
import torch
import transformers

from exceptions import raise_for_status
from ffmpeg_util import *

# from accelerate import cpu_offload_with_hook

try:
    from diffusers import ConfigMixin
except ImportError:
    ConfigMixin = None


DEVICE_ID = os.environ.get("DEVICE_ID", "").strip() or "cuda:0"
SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()
DISABLE_CPU_OFFLOAD = bool(
    int(os.environ.get("DISABLE_CPU_OFFLOAD", "").strip() or "0")
)
CHECKPOINTS_DIR = (
    os.environ.get("CHECKPOINTS_DIR", "").strip()
    or "/root/.cache/gooey-gpu/checkpoints"
)

try:
    gpu_limit_gib = float(os.environ["RESOURCE_LIMITS_GPU"].replace("Gi", ""))
except (KeyError, ValueError):
    print("RESOURCE_LIMITS_GPU environment variable not set to a valid value.")
else:
    total_mem_bytes = torch.cuda.mem_get_info()[1]
    fraction = gpu_limit_gib * 1024**3 / total_mem_bytes
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"GPU memory limit set to {gpu_limit_gib}Gi ({fraction:.2%})")


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


def endpoint(fn):
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        for k, t in signature.parameters.items():
            model = t.annotation
            if issubclass(model, BaseModel):
                kwargs[k] = model.parse_obj(kwargs[k])
        print(f"---> {fn.__name__}: {kwargs!r}")
        try:
            ret = fn(*args, **kwargs)
            if isinstance(ret, BaseModel):
                ret = ret.dict()
            return ret
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return wrapper


M = typing.TypeVar(
    "M", bound=typing.Union[torch.nn.Module, ConfigMixin, transformers.Pipeline]
)


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


def download_images(
    urls: typing.List[str],
    max_size: typing.Tuple[int, int] = None,
    mode: str = "RGB",
) -> typing.List[PIL.Image.Image]:
    def fn(url):
        return download_image(url, max_size, mode)

    return list(map_parallel(fn, urls))


def download_image(
    url: str,
    max_size: typing.Tuple[int, int] = None,
    mode: str = "RGB",
) -> PIL.Image.Image:
    im_bytes = requests.get(url).content
    f = io.BytesIO(im_bytes)
    im_pil = PIL.Image.open(f).convert(mode)
    if max_size:
        im_pil = downscale_img(im_pil, max_size)
    return im_pil


def downscale_img(
    im_pil: PIL.Image.Image, max_size: typing.Tuple[int, int]
) -> PIL.Image.Image:
    downscale_factor = get_downscale_factor(im_size=im_pil.size, max_size=max_size)
    if downscale_factor:
        im_pil = PIL.ImageOps.scale(im_pil, downscale_factor)
        print(f"Resize image by {downscale_factor:.3f}x = {im_pil.size}")
    return im_pil


def get_downscale_factor(
    *, im_size: typing.Tuple[int, int], max_size: typing.Tuple[int, int]
) -> typing.Optional[float]:
    downscale_factor = math.sqrt(
        (max_size[0] * max_size[1]) / (im_size[0] * im_size[1])
    )
    if downscale_factor < 0.99:
        return downscale_factor
    else:
        return None


def upload_images(images: typing.List[PIL.Image.Image], upload_urls: typing.List[str]):
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
    raise_for_status(r)


def apply_parallel(fn, *iterables):
    threads = [threading.Thread(target=fn, args=args) for args in zip(*iterables)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def map_parallel(fn, it):
    with ThreadPoolExecutor(max_workers=len(it)) as pool:
        return list(pool.map(fn, it))


def upload_audio(audio, url: str, rate: int = 16_000):
    import scipy

    # The resulting audio output can be saved as a .wav file:
    f = io.BytesIO()
    scipy.io.wavfile.write(f, rate=rate, data=audio)
    audio_bytes = f.getvalue()
    # upload to given url
    upload_audio_from_bytes(audio_bytes, url)


def upload_audio_from_bytes(audio: bytes, url: str):
    r = requests.put(url, headers={"Content-Type": "audio/wav"}, data=audio)
    raise_for_status(r)


def upload_video_from_bytes(video, url: str):
    r = requests.put(url, headers={"Content-Type": "video/mp4"}, data=video)
    raise_for_status(r)


# Add some missing mimetypes
mimetypes.add_type("audio/wav", ".wav")


def download_file_to_path(*, url: str, path: str, cached: bool = False):
    if cached and os.path.exists(path):
        return
    r = requests.get(url)
    raise_for_status(r, is_user_url=not cached)
    with open(path, "wb") as f:
        f.write(r.content)
