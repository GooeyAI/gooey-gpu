import os
import traceback
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
    # task_acks_late=True,
)


init_fns = []


@worker_init.connect()
def init_all(**kwargs):
    for fn in init_fns:
        fn(**kwargs)


def setup_queues(
    *,
    model_ids: typing.List[str],
    load_fn: typing.Callable[[str], typing.Any],
    queue_prefix: str = os.environ.get("QUEUE_PREFIX", "gooey-gpu"),
):
    def init(**kwargs):
        for model_id in model_ids:
            try:
                load_fn(model_id)
            except Exception as e:
                traceback.print_exc()
                raise

    init_fns.append(init)

    app.conf.task_queues = app.conf.task_queues or []
    for model_id in model_ids:
        queue = os.path.join(queue_prefix, model_id).strip("/")
        app.conf.task_queues.append(Queue(queue))
