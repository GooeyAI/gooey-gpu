# Autoscaling with RabbitMQ based on queue-length

Done with KEDA (https://learnk8s.io/scaling-celery-rabbitmq-kubernetes).

## Install KEDA

`kubectl apply --server-side -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml`

## Add autoscaling entry with deployment name

In `rabbitmqAutoscaling` in [`chart/values.yaml`](chart/values.yaml).

e.g.

```yaml
rabbitmqAutoscaling:
  enabled: true
  deployments:
    - name: "deforum-sd-2"
      queueName: "gooey-gpu/Protogen_V2.2.ckpt"
      queueLength: "2"  # tasks per replica
      minReplicaCount: 2
      maxReplicaCount: 12
```

`queueLength` is the number of tasks you want to have per replica of this deployment.

## Debugging

* `kubectl get scaledobject` -- run `kubectl describe scaledobject <name>` for more info on it
* `kubectl get hpa` -- An HPA must be created for every scaled object
* `kubectl get pods -n keda` -- inspect helper pods in keda namespace
* `kubectl describe deployment <name>` -- for more info on its scaling
