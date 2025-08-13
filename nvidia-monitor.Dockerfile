FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /root
COPY scripts/monitor.sh monitor.sh

# ./monitor.sh
# nvidia-smi
