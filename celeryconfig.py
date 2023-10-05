import os
import typing

from celery import Celery
from celery.signals import worker_init
from kombu import Queue

app = Celery()

app.conf.update(
    broker_url=os.environ["BROKER_URL"],
    result_backend=os.environ["RESULT_BACKEND"],
    imports=os.environ["IMPORTS"].split(),
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_acks_late=True,
)


def setup_queues(
    *,
    model_ids: typing.List[str],
    load_fn: typing.Callable[[str], None],
    queue_prefix: str = os.environ.get("QUEUE_PREFIX", "gooey-gpu"),
):
    @worker_init.connect()
    def init(**kwargs):
        for model_id in model_ids:
            load_fn(model_id)

    app.conf.task_queues = app.conf.task_queues or []
    for model_id in model_ids:
        queue = os.path.join(queue_prefix, model_id).strip("/")
        app.conf.task_queues.append(Queue(queue))
