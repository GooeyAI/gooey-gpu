# k8s on azure

## Creating a Nodepool

For example,

```
az aks nodepool add --name t4 --resource-group prod --node-vm-size standard_nc64as_t4_v3 --cluster-name gooey-prod --enable-cluster-autoscaler --min-count 0 --max-count 5
```

To add node taint, use `--node-taints sku=gpu:NoSchedule`

## To add Nvidia Extension

https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux

## To enable autoscaling based on rabbitmq queue length

https://ryanbaker.io/2019-10-07-scaling-rabbitmq-on-k8s/
