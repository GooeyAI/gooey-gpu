FROM nvcr.io/nvidia/cuda:12.5.0-runtime-ubuntu22.04

# "https://github.com/containerd/nerdctl/releases/download/v1.7.5/nerdctl-full-1.7.5-linux-amd64.tar.gz" \
RUN wget -qO nerdctl.tar.gz "https://github.com/containerd/nerdctl/releases/download/v1.7.6/nerdctl-1.7.6-linux-amd64.tar.gz" \
    && tar Cxzvvf /usr/local nerdctl.tar.gz \
    && rm nerdctl.tar.gz

# for i in $(nerdctl  -a /host/run/containerd/containerd.sock -n k8s.io container ls --format "{{.ID}}"); do nerdctl  -a /host/run/containerd/containerd.sock -n k8s.io inspect -f '{{.State.Pid}} {{index .Config.Labels "io.kubernetes.pod.name"}}' $i; done | grep gooey-gpu
# nvidia-smi
