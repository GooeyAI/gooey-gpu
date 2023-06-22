import os

from celery import Celery

app = Celery()

app.conf.update(
    broker_url=os.environ["BROKER_URL"],
    result_backend=os.environ["RESULT_BACKEND"],
    imports=os.environ["IMPORTS"].split(),
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_acks_late=True,
)
