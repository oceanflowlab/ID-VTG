#!/usr/bin/env bash
set -u

BASE="experiments/idvtg_gym"
MODELDIR="$BASE/models"
OUTDIR="$BASE/eval"
mkdir -p "$OUTDIR"

# 提取 mIoU 的函数
get_score_from_txt() {
  local txtfile="$1"
  grep -oP 'MR-full-mIoU:\s*\K[0-9]+(\.[0-9]+)?' "$txtfile" | tail -n 1
}

# 初始化最优记录
best_miou=-1
best_i=-1

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

    logfile="$OUTDIR/eval_best${i}.log"
    CUDA_VISIBLE_DEVICES=$k python eval.py --name idvtg_gym --opt idvtg_gym_opt_all --ckpt best${i} --vis True > "$logfile" 2>&1

    # 提取本次评估的 mIoU
    miou=$(get_score_from_txt "$logfile")
    if [ -n "$miou" ]; then
      echo "$(date): extracted mIoU = $miou for best${i}"
      # 使用 awk 进行浮点数比较
      if awk -v m="$miou" -v b="$best_miou" 'BEGIN{exit !(m > b)}'; then
        echo "$(date): new best mIoU ($miou > $best_miou), updating..."
        # 删除旧的最优模型（如果存在）
        if [ $best_i -ne -1 ]; then
          old_ckpt="$MODELDIR/best${best_i}.pth"
          if [ -f "$old_ckpt" ]; then
            echo "$(date): removing old best checkpoint $old_ckpt"
            rm -f "$old_ckpt"
          fi
        fi
        best_miou=$miou
        best_i=$i
        echo "$(date): current best mIoU = $best_miou (from best${best_i})"
      else
        echo "$(date): mIoU $miou not greater than current best $best_miou, keep both"
      fi
    else
      echo "$(date): warning: failed to extract mIoU from $logfile"
    fi

    i=$((i+1))
  else
    sleep 600  # 10 minutes
  fi
done