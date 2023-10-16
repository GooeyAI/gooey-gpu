# Monitoring

Default GKE metrics and logging is enabled.

CPU usage, memory usage, network usage, logs, etc. from the kubernetes cluster
are already being sent to the GKE dashboard in Google Cloud.

These can also be used to configure alerts.

## Prometheus

Besides that, we also have a prometheus for arbitrary metrics.

We use the `kube-prometheus-stack` helm chart, along with some other charts for
exporters.

### Setting up `kube-prometheus-stack` in a new cluster

This only needs to be done once in a cluster.

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prom-stack monitoring-chart/ --debug
```

### Values

Get available configuration options:

```bash
helm show values prometheus-community/kube-prometheus-stack
```

The chart configuration for kube-prometheus-stack is under the `kube-prometheus-stack` key
in [`monitoring-chart/values.yaml`](/monitoring-chart/values.yaml).

### To apply changes to values

Update [monitoring-chart/values.yaml](/monitoring-chart/values.yaml) and run ->

```bash
helm upgrade prom-stack monitoring-chart/ --debug
```
