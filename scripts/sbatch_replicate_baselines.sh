#!/bin/bash
#SBATCH -J qmix-baselines
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/gcsl_mas/logs/replicate_baselines_%j.log
# ============================================================
#  Baselines for the paper's Table 1 on GPT-OSS-120B: AutoGen, LangGraph, Agent Framework and
#  Lobster through agent_baseline/run_benchmark.py, plus single-agent Base.
#  Frameworks install from PyPI into gcsl_mas/.venv_base rather than from the vendored copies.
#
#  Idempotent: finished framework x benchmark pairs are skipped.
#  Submit from a login node, from the Agent-Q-Mix repo root:
#    sbatch scripts/sbatch_replicate_baselines.sh
# ============================================================
set -uo pipefail

GCSL=/work/11138/pranavbelligundu/vista/gcsl_mas
REPO=$GCSL/Agent-Q-Mix
MODEL=${MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8234}
FRAMEWORKS=${FRAMEWORKS:-"lobster autogen langgraph agent-framework"}
BENCHES=${BENCHES:-"mmlu aime_2025 aime_2026 hmmt_feb_2025 beyond_aime humaneval livecodebench"}
BASE_BENCHES=${BASE_BENCHES:-"mmlu_pro aime_2025 aime_2026 hmmt_2025 beyond_aime humaneval livecodebench"}
EVAL_LIMIT=${EVAL_LIMIT:-}     # for tests only
OUT_ROOT=${OUT_ROOT:-/scratch/11138/pranavbelligundu/gcsl_mas/replicate_qmix}
OUT=$OUT_ROOT/baselines

export HF_HOME=/scratch/11138/pranavbelligundu/hf_home
export VLLM_USE_FLASHINFER_SAMPLER=0
export BASE_URL=http://localhost:$PORT/v1 API_KEY=EMPTY LLM_MODEL=$MODEL

mkdir -p "$OUT" /scratch/11138/pranavbelligundu/gcsl_mas/logs
cd "$REPO"
echo "[base] host=$(hostname) job=${SLURM_JOB_ID:-none} model=$MODEL frameworks=$FRAMEWORKS commit=$(git rev-parse --short HEAD)"

VLOG="$OUT/vllm_${SLURM_JOB_ID:-local}.log"
"$GCSL/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" --served-model-name "$MODEL" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
ready=0
for i in $(seq 1 360); do
    curl -sf "localhost:$PORT/v1/models" >/dev/null && { ready=1; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "[base] vLLM died at startup; see $VLOG"; tail -20 "$VLOG"; exit 1; }
    sleep 5
done
[ $ready = 1 ] || { echo "[base] vLLM not ready after 30 min"; exit 1; }
echo "[base] vLLM ready after $((i * 5))s"

bench_key() { case $1 in mmlu) echo mmlu_pro ;; hmmt_feb_2025) echo hmmt_2025 ;; *) echo "$1" ;; esac; }
split_of() { case $1 in mmlu_pro) echo validation ;; aime_2025|aime_2026|hmmt_2025) echo train ;; *) echo test ;; esac; }
limit_args=(); [ -n "$EVAL_LIMIT" ] && limit_args=(--limit "$EVAL_LIMIT")

pids=()
for f in $FRAMEWORKS; do
    for b in $BENCHES; do
        out=$OUT/${f}_$(bench_key "$b").json
        if [ -s "$out" ]; then echo "[base] $f $b: done"; continue; fi
        ( cd "$REPO/agent_baseline" && PYTHONPATH=$PWD "$GCSL/.venv_base/bin/python" -W ignore run_benchmark.py \
            -f "$f" -b "$b" ${EVAL_LIMIT:+-l "$EVAL_LIMIT"} -o "$OUT" > "$OUT/${f}_$b.log" 2>&1 ) &
        pids+=($!)
    done
done
for b in $BASE_BENCHES; do
    out=$OUT/base_$b.json
    if [ -s "$out" ]; then echo "[base] base $b: done"; continue; fi
    PYTHONPATH=$REPO "$GCSL/.venv_qmix/bin/python" -W ignore -m experiments.run_base_eval --dataset "$b" \
        --split "$(split_of "$b")" --llm_name "$MODEL" "${limit_args[@]}" --output_path "$out" > "$OUT/base_$b.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

python3 "$REPO/experiments/summarize_replication.py" "$OUT_ROOT"
