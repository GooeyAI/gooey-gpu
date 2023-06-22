Create a cluster - https://console.cloud.google.com/kubernetes/add

Add gpu node pool

Install nvidia drivers (newer version) - https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers

Deploy rabbitmq - https://console.cloud.google.com/kubernetes/list/overview?project=dara-c1b52

Deploy redis - https://console.cloud.google.com/marketplace/kubernetes/config/google/redis-ha?version=7.0&project=dara-c1b52

create your `values.yaml` using the example `values.example.yaml`

Deploy using helm - 
```bash
helm install gooey-gpu-1 chart/ -f values.yaml
```

To access this outside the k8s cluster, but from another GCP VM inside the same VPC, 
create internal passthrough load balancer services for rabbitmq and redis - https://cloud.google.com/kubernetes-engine/docs/how-to/internal-load-balancing#subsetting
[ilb_svc.yaml](/k8s/ilb_svc.yaml)

To access this from your dev machine us k8s port forwarding - https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster
(replace `rabbitmq-1-rabbitmq-0` with whatever `kubectl get pod` shows for rabbitmq & same for redis)
```
kubectl port-forward rabbitmq-1-rabbitmq-0 5672:5672
kubectl port-forward redis-ha-1-server-0 6379:6379
```
