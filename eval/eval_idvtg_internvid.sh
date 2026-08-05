#!/usr/bin/env bash
set -u

BASE="experiments/idvtg_internvid"
MODELDIR="$BASE/models"
OUTDIR="$BASE/eval"
mkdir -p "$OUTDIR"     
i=2


while true; do
  if [ -f "$MODELDIR/best${i}.pth" ]; then
    echo "$(date): found best${i}.pth"

    # 找到第一个空闲内存 > 3000 MiB 的 GPU
    while true; do
      if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found, retry in 60s"
        sleep 60
        continue
      fi
      mapfile -t frees < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
      found=-1
      for idx in "${!frees[@]}"; do
        free="$(echo "${frees[$idx]}" | tr -d '[:space:]')"
        if [ -n "$free" ] && [ "$free" -gt 3000 ]; then
          found=$idx
          break
        fi
      done

      if [ "$found" -ge 0 ]; then
        k=$found
        echo "$(date): using GPU $k (free ${frees[$k]} MiB)"
        break
      else
        echo "$(date): no GPU with >3000MiB free, retry in 60s"
        sleep 60
      fi
    done

    CUDA_VISIBLE_DEVICES=$k python eval.py --name idvtg_internvid --opt idvtg_internvid_opt_all --ckpt best${i} --vis True > "$OUTDIR/eval_best${i}.log" 2>&1

    i=$((i+1))
  else
    sleep 600  # 10 minutes
  fi
done
