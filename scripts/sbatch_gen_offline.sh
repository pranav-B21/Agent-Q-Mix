#!/bin/bash
#SBATCH -J gcsl-gen-offline
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/gcsl_mas/logs/gen_offline_%j.log
# ============================================================
#  Offline behavior-policy dataset for Setting B (goal-conditioned topology controller).
#  One GH200 node: a local vLLM server for Qwen3-1.7B + NUM_SHARDS generator processes
#  (uniform-random topologies, epsilon=1), one JSONL per shard.
#
#  Idempotent: resubmitting resumes every shard from its JSONL.
#  Submit from a login node, from the Agent-Q-Mix repo root:
#    sbatch scripts/sbatch_gen_offline.sh
#    OUT_DIR=... DATASETS="mmlu_pro:300,math500_int:311" REPEATS=3 sbatch scripts/sbatch_gen_offline.sh
# ============================================================
set -uo pipefail

GCSL=/work/11138/pranavbelligundu/vista/gcsl_mas
REPO=$GCSL/Agent-Q-Mix
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
PORT=${PORT:-8234}
DATASETS=${DATASETS:-mmlu_pro:300,math500_int:311}
REPEATS=${REPEATS:-3}
SEED=${SEED:-0}
NUM_SHARDS=${NUM_SHARDS:-24}
OUT_DIR=${OUT_DIR:-/scratch/11138/pranavbelligundu/gcsl_mas/offline_v1}   # v0 predates the graph.py fix; never resume into it
export QMIX_TASK_ENCODER=${QMIX_TASK_ENCODER:-st}

export HF_HOME=/scratch/11138/pranavbelligundu/hf_home
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer's sampler JIT needs ninja+nvcc, absent here
export PYTHONPATH=$REPO

mkdir -p "$OUT_DIR" /scratch/11138/pranavbelligundu/gcsl_mas/logs
cd "$REPO"
echo "[gen] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT_DIR datasets=$DATASETS repeats=$REPEATS shards=$NUM_SHARDS"

"$GCSL/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" --served-model-name "$MODEL" \
    --gpu-memory-utilization 0.85 --max-model-len 16384 > "$OUT_DIR/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 900); do
    curl -sf "localhost:$PORT/v1/models" >/dev/null && break
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[gen] vLLM died at startup; see $OUT_DIR/vllm_${SLURM_JOB_ID:-local}.log"; exit 1
    fi
    sleep 1
done
curl -sf "localhost:$PORT/v1/models" >/dev/null || { echo "[gen] vLLM not ready after 15 min"; exit 1; }
echo "[gen] vLLM ready after ${i}s"

# The .env in the repo points at localhost:8234; override in case PORT differs.
export BASE_URL=http://localhost:$PORT/v1 API_KEY=EMPTY

for s in $(seq 0 $((NUM_SHARDS - 1))); do
    sid=$(printf "%02d" "$s")
    "$GCSL/.venv_qmix/bin/python" -W ignore -m experiments.gen_offline_data \
        --datasets "$DATASETS" --repeats "$REPEATS" --seed "$SEED" --llm_name "$MODEL" \
        --shard_id "$s" --num_shards "$NUM_SHARDS" \
        --out "$OUT_DIR/shard_$sid.jsonl" >> "$OUT_DIR/shard_$sid.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$VLLM_PID$")

n=$(cat "$OUT_DIR"/shard_*.jsonl 2>/dev/null | wc -l)
echo "[gen] done: $n episodes in $OUT_DIR"
"$GCSL/.venv_qmix/bin/python" - "$OUT_DIR" <<'EOF'
import json, glob, sys, collections
recs = [json.loads(l) for f in glob.glob(sys.argv[1] + "/shard_*.jsonl") for l in open(f)]
by = collections.defaultdict(list)
for r in recs:
    by[r["dataset"]].append(r)
for d, rs in sorted(by.items()):
    acc = sum(r["accuracy"] for r in rs) / len(rs)
    tok = sorted(r["total_tokens"] for r in rs)
    print(f"[gen] {d}: n={len(rs)} acc={acc:.3f} tokens p10/p50/p90={tok[len(tok)//10]}/{tok[len(tok)//2]}/{tok[9*len(tok)//10]}")
EOF
