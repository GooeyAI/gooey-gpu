#!/bin/bash

set -e

ENV_VAR_FILTER="MODEL_IDS"

function print3() {
  printf "%-8s %-80s %12s\n" "$1" "${2%,}" "$3"
}

print3 "PID" "Model IDs" "GPU Mem"

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    | while IFS=',' read -r pid mem; do
        pid="${pid//[[:space:]]/}"
        mem="$(echo $mem|tr -d '[:space:],')"
        cmd_env=$(ps eww -p "$pid" -o command=)
        val=$(printf '%s\n' "$cmd_env" | tr ' ' '\n' | grep "$ENV_VAR_FILTER=" | cut -d= -f2- | tr '\n' ',')
        if [ -z "$val" ]; then
          val=$(printf '%s\n' "$cmd_env" | sed -nE 's/.*celery@([^:]+):MainProcess.*/\1/p')
        fi
        mem_gib=$(awk "BEGIN{printf \"%.2f\", $mem/1024}")
        print3 "$pid" "${val%,}" "${mem_gib}GiB"
      done
