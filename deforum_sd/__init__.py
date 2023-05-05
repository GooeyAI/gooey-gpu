import shutil
import uuid

import requests
from fastapi import APIRouter
from starlette.requests import Request

import gooey_gpu
from api import PipelineInfo
from deforum_sd import deforum_script

app = APIRouter()


@app.post("/deforum/")
@gooey_gpu.endpoint
def deforum(
    request: Request, pipeline: PipelineInfo, inputs: deforum_script.DeforumAnimArgs
):
    # init args
    args = deforum_script.DeforumArgs(batch_name=str(uuid.uuid1()))
    args.seed = pipeline.seed
    if pipeline.scheduler:
        args.sampler = pipeline.scheduler
    anim_args = deforum_script.DeforumAnimArgs()
    for k, v in inputs.dict().items():
        setattr(anim_args, k, v)
    try:
        # run inference
        args, anim_args = run_deforum(pipeline=pipeline, args=args, anim_args=anim_args)
        # generate video
        vid_path = deforum_script.create_video(args, anim_args)
        with open(vid_path, "rb") as f:
            vid_bytes = f.read()
    finally:
        # cleanup
        shutil.rmtree(args.outdir, ignore_errors=True)
    # upload videos
    for url in pipeline.upload_urls:
        r = requests.put(
            url,
            headers={"Content-Type": "video/mp4"},
            data=vid_bytes,
        )
        r.raise_for_status()
        return


@gooey_gpu.gpu_task
def run_deforum(*, pipeline: PipelineInfo, args, anim_args):
    root = load_deforum(pipeline)
    # with gooey_gpu.use_models(root.model):
    deforum_script.run(root, args, anim_args)
    return args, anim_args


_deforum_cache = {}


def load_deforum(pipeline):
    try:
        root = _deforum_cache[pipeline.model_id]
    except KeyError:
        root = deforum_script.Root()
        root.map_location = gooey_gpu.DEVICE_ID
        root.model_checkpoint = pipeline.model_id
        deforum_script.setup(root)
        _deforum_cache[pipeline.model_id] = root
        root.model.to(gooey_gpu.DEVICE_ID)
    return root
