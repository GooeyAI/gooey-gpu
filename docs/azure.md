# Azure

## Installation

Copy `values.example.yaml` to `values-azure.yaml` and set the
values correctly.

```
# first time
helm install gooey-gpu chart -f chart/values-azure.yaml -f values-azure.yaml --debug

# to upgrade
helm upgrade gooey-gpu chart -f chart/values-azure.yaml -f values-azure.yaml --debug
```

The values in `chart/values-azure.yaml` will overwrite the
default `chart/values.yaml`. This is needed because the
deployments differ between Azure and GCP.

To move a deployment from GCP to Azure, the same deployment
values can be used, with changes to the `nodeSelector` to
target the correct kind of node.
