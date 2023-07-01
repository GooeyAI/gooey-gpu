# Cluster Setup and Deployments

Follow these steps to create a cluster, add GPU nodes, install NVIDIA drivers, deploy RabbitMQ and Redis, and access deployments.

## Create a Cluster

1. Visit [Google Cloud Console](https://console.cloud.google.com/kubernetes/add) and create a cluster.

## Add GPU Node Pool

2. Add a GPU node pool to your cluster.

## Install NVIDIA Drivers (newer version)

3. Install NVIDIA drivers in your cluster by following the [official guide](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers).

## Deploy RabbitMQ

4. Deploy RabbitMQ using the [Google Cloud Marketplace for RabbitMQ](https://console.cloud.google.com/marketplace/kubernetes/config/google/rabbitmq).

## Deploy Redis

5. Deploy Redis by using [Google Cloud Marketplace for Redis](https://console.cloud.google.com/marketplace/kubernetes/config/google/redis-ha).

## Create Values.yaml

6. Create your `values.yaml` file using the example `values.example.yaml`.

## Deploy Using Helm

7. Use Helm to deploy with the following command:

```bash
helm install gooey-gpu-1 chart/ -f values.yaml
```

---

## Access Deployments

### Inside VPC (GCP VM)

- Create internal passthrough load balancer services for RabbitMQ and Redis by following the [official guide on subsetting](https://cloud.google.com/kubernetes-engine/docs/how-to/internal-load-balancing#subsetting).
- Use the provided `ilb_svc.yaml` file: [ilb_svc.yaml](/k8s/ilb_svc.yaml)

### From Your Dev Machine

- Access deployments from your dev machine using Kubernetes port forwarding as described in the [official documentation](https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster).

- Replace `rabbitmq-1-rabbitmq-0` and `redis-ha-1-server-0` with whatever `kubectl get pod` shows for RabbitMQ and Redis.

```
kubectl port-forward rabbitmq-1-rabbitmq-0 15674:15672 5674:5672 & kubectl port-forward redis-ha-1-server-0 6374:6379
```
