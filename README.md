# gooey-gpu

Docker container which provides inference endpoints for the following - 

Endpoint | Models
-- | --
`/text2img/` |  `StableDiffusionPipeline` on huggingface
`/img2img/`  |  `StableDiffusionImg2ImgPipeline` on huggingface
`/inpaint/`  |  `StableDiffusionInpaintPipeline` on huggingface
`/upscale/`  |  `StableDiffusionUpscalePipeline` on huggingface
`/instruct_pix2pix/` |  `StableDiffusionInstructPix2PixPipeline` on huggingface
`/deforum/` | [deforum-stable-diffusion](https://github.com/deforum-art/deforum-stable-diffusion)
`/image-captioning/` | [LAVIS image captioning](https://github.com/salesforce/LAVIS/#image-captioning)
`/vqa/` | [LAVIS visual Q&A](https://github.com/salesforce/LAVIS/#visual-question-answering-vqa)

---

Models are automatically offloaded to cpu ram after inference. 

This allows you to run this container with an idle GPU usage of just ~2GB

---

E.g. 

```
# starts the docker container on port 5012
./scripts/run-dev.sh  
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
    "http://localhost:5012/text2img/",
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
