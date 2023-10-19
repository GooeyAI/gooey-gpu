import os

from celery.signals import worker_init
from kombu import Queue

import gooey_gpu
from api import PipelineInfo, WhisperInputs, AsrOutput
from celeryconfig import app

QUEUE_PREFIX = os.environ.get("QUEUE_PREFIX", "gooey-gpu")
MODEL_IDS = os.environ["TEXT_EMBED_MODEL_IDS"].split()

app.conf.task_queues = app.conf.task_queues or []
for model_id in MODEL_IDS:
    queue = os.path.join(QUEUE_PREFIX, model_id).strip("/")
    app.conf.task_queues.append(Queue(queue))


@worker_init.connect()
def init(**kwargs):
    # app.conf.task_queues = []
    for model_id in MODEL_IDS:
        load_pipe(model_id)


@app.task(name="embeddings")
@gooey_gpu.endpoint
def text_embed(pipeline: PipelineInfo, inputs: WhisperInputs) -> AsrOutput:
    pass
