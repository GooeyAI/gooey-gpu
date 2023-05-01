# gooey-gpu

Docker container which provides inference endpoints for the following - 

Endpoint | Models
--- | ---
`/text2img/` |  `StableDiffusionPipeline` [models](https://huggingface.co/models?sort=downloads&search=diffusion)
`/img2img/`  |  `StableDiffusionImg2ImgPipeline` [models](https://huggingface.co/models?sort=downloads&search=diffusion)
`/inpaint/`  |  `StableDiffusionInpaintPipeline` [models](https://huggingface.co/models?sort=downloads&search=inpainting)
`/upscale/`  |  `StableDiffusionUpscalePipeline` [models](https://huggingface.co/models?sort=downloads&search=upscaler)
`/instruct_pix2pix/` |  [`StableDiffusionInstructPix2PixPipeline`](https://huggingface.co/timbrooks/instruct-pix2pix)
`/deforum/` | [deforum-stable-diffusion](https://github.com/deforum-art/deforum-stable-diffusion)
`/vqa/` | [Visual question answering](https://github.com/salesforce/LAVIS#visual-question-answering-vqa)
`/image-captioning/` | [LAVIS image captioning](https://github.com/salesforce/LAVIS/#image-captioning)
`/controlnet/`  | [`StableDiffusionControlNetPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet) using [`ControlNetModel`](https://huggingface.co/lllyasviel?q=controlnet)
`/whisper/` | [transformers asr pipeline](https://huggingface.co/docs/transformers/tasks/asr)
`/nemo/asr/` | [Nvidia Nemo ASR](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html)
`/deepfloyd_if/text2img/` | [DeepFloyd IF](https://github.com/deep-floyd/IF)
---

Features: 
 - Models are automatically offloaded to cpu ram after inference. This allows you to run this container with an idle GPU usage of just ~2GB
 - Error handling via sentry
 - Multiple workers support
 - Uploads images to given upload url via POST, useful for directly uploading to google cloud storage etc.
 - Error pass-through to the clients, via HTTP 500 response body

---

E.g. 

```
# starts the docker container on port 5012
./scripts/run-dev.sh common
```

View API docs at `http://localhost:5012/docs`

```python
import requests
from firebase_admin import storage

# upload the result to google cloud storage
blob = storage.bucket("my-gcs-bucket").blob("path/to/img.png")
upload_url = blob.generate_signed_url(
    version="v4",
    # This URL is valid for 15 minutes
    expiration=datetime.timedelta(minutes=30),
    # Allow PUT requests using this URL.
    method="PUT",
    content_type="image/png",
)

# run stable diffusion 2.1
r = requests.post(
    "http://localhost:5000/text2img/",
    json=dict(
        pipeline={
            "upload_urls": [upload_url],
            "model_id": "stabilityai/stable-diffusion-2-1",
            "seed": 42,
        },
        inputs={
            "prompt": ["A furry friend"],
            "num_images_per_prompt": 1,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 768,
            "height": 768,
        },
    )
)
r.raise_for_status()

print(blob.public_url)
```
