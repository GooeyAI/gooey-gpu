import os
import shutil
import uuid
from functools import lru_cache

import requests

import gooey_gpu
from api import PipelineInfo
from celeryconfig import app, setup_queues
from deforum_sd import deforum_script


@app.task(name="deforum")
@gooey_gpu.endpoint
def deforum(pipeline: PipelineInfo, inputs: deforum_script.DeforumAnimArgs):
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
        root = load_deforum(pipeline.model_id)
        deforum_script.run(root, args, anim_args)
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


@lru_cache
def load_deforum(chkpt: str):
    print(f"Loading deforum model {chkpt!r}...")
    root = deforum_script.Root()
    root.map_location = gooey_gpu.DEVICE_ID
    root.model_checkpoint = chkpt
    deforum_script.setup(root)
    root.model.to(gooey_gpu.DEVICE_ID)
    return root


setup_queues(
    model_ids=os.environ["DEFORUM_MODEL_IDS"].split(),
    load_fn=load_deforum,
)
