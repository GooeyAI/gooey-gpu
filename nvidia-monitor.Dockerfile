FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN wget -qO nerdctl.tar.gz "https://github.com/containerd/nerdctl/releases/download/v1.7.6/nerdctl-1.7.6-linux-amd64.tar.gz" \
    && tar xvzf /usr/local/bin nerdctl.tar.gz \
    && rm nerdctl.tar.gz

# for i in $(nerdctl  -a /host/run/containerd/containerd.sock -n k8s.io container ls --format "{{.ID}}"); do nerdctl  -a /host/run/containerd/containerd.sock -n k8s.io inspect -f '{{.State.Pid}} {{index .Config.Labels "io.kubernetes.pod.name"}}' $i; done | grep gooey-gpu
# nvidia-smi
