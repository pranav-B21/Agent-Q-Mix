#!/bin/bash
#SBATCH -J qmix-replicate
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/gcsl_mas/logs/replicate_qmix_%j.log
# ============================================================
#  Replicate the Agent Q-Mix row of the paper's Table 1 on GPT-OSS-120B, with the paper's recipe
#  (Appendix D, Table 5): 3 per-domain controllers, 50 episodes, 15 examples, batch 8,
#  w_acc 1.25, w_tok 0.10, 3 rounds for math and 2 otherwise, median of 3 runs.
#
#  VARIANT=released  upstream execution and training loop, bug included (QMIX_LEGACY_EXEC=1)
#  VARIANT=fixed     dependency-ordered execution; agents actually see each other's answers
#
#  Idempotent: finished controllers and benchmarks are skipped, so resubmitting continues. A
#  benchmark cut off partway restarts from its first item.
#  Submit from a login node, from the Agent-Q-Mix repo root:
#    VARIANT=released sbatch scripts/sbatch_replicate_qmix.sh
#    VARIANT=fixed    sbatch scripts/sbatch_replicate_qmix.sh
# ============================================================
set -uo pipefail

VARIANT=${VARIANT:?set VARIANT=released or VARIANT=fixed}
case $VARIANT in released|fixed) ;; *) echo "VARIANT must be released or fixed"; exit 1 ;; esac
GCSL=/work/11138/pranavbelligundu/vista/gcsl_mas
REPO=$GCSL/Agent-Q-Mix
MODEL=${MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8234}
RUNS=${RUNS:-"1 2 3"}
BENCHES=${BENCHES:-"mmlu_pro aime_2025 aime_2026 hmmt_2025 beyond_aime humaneval livecodebench"}
EPISODES=${EPISODES:-50}
DATA_LIMIT=${DATA_LIMIT:-15}
EVAL_LIMIT=${EVAL_LIMIT:-}     # for tests only
OUT_ROOT=${OUT_ROOT:-/scratch/11138/pranavbelligundu/gcsl_mas/replicate_qmix}
OUT=$OUT_ROOT/$VARIANT

export HF_HOME=/scratch/11138/pranavbelligundu/hf_home
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer's sampler JIT needs ninja+nvcc, absent here
export PYTHONPATH=$REPO
export BASE_URL=http://localhost:$PORT/v1 API_KEY=EMPTY
[ "$VARIANT" = released ] && export QMIX_LEGACY_EXEC=1 || export QMIX_LEGACY_EXEC=0

PY="$GCSL/.venv_qmix/bin/python -W ignore"
mkdir -p "$OUT" /scratch/11138/pranavbelligundu/gcsl_mas/logs
cd "$REPO"
echo "[rep] host=$(hostname) job=${SLURM_JOB_ID:-none} variant=$VARIANT model=$MODEL runs=$RUNS benches=$BENCHES episodes=$EPISODES commit=$(git rev-parse --short HEAD)"

VLOG="$OUT/vllm_${SLURM_JOB_ID:-local}.log"
"$GCSL/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" --served-model-name "$MODEL" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
ready=0
for i in $(seq 1 360); do
    curl -sf "localhost:$PORT/v1/models" >/dev/null && { ready=1; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "[rep] vLLM died at startup; see $VLOG"; tail -20 "$VLOG"; exit 1; }
    sleep 5
done
[ $ready = 1 ] || { echo "[rep] vLLM not ready after 30 min"; exit 1; }
echo "[rep] vLLM ready after $((i * 5))s"

domain_of() { case $1 in mmlu_pro) echo reasoning ;; humaneval|livecodebench) echo coding ;; *) echo math ;; esac; }
rounds_of() { [ "$1" = math ] && echo 3 || echo 2; }
split_of() { case $1 in mmlu_pro) echo validation ;; aime_2025|aime_2026|hmmt_2025) echo train ;; *) echo test ;; esac; }
train_data() { case $1 in reasoning) echo mmlu_pro ;; coding) echo livecodebench_testgen ;; math) echo aime_2024 ;; esac; }
train_split() { case $1 in reasoning) echo validation ;; coding) echo test ;; math) echo train ;; esac; }
has_results() { [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$1'))['metrics']['total']>0 else 1)" 2>/dev/null; }

# ---- Train: 3 controllers per run. Paper's Table 4 trains MMLU-Pro on its validation split.
pids=()
for r in $RUNS; do
    for d in reasoning coding math; do
        dir=$OUT/run$r/ckpt; mkdir -p "$dir"
        if [ -f "$dir/$d.done" ]; then echo "[rep] run$r $d: controller done"; continue; fi
        (
            $PY -m experiments.run_qmix_train --dataset "$(train_data $d)" --split "$(train_split $d)" \
                --llm_name "$MODEL" --num_episodes "$EPISODES" --num_rounds "$(rounds_of $d)" \
                --batch_size 8 --data_limit "$DATA_LIMIT" --token_penalty 0.10 --accuracy_weight 1.25 \
                --device cpu --seed "$r" --save_path "$dir/$d.pt" > "$dir/$d.log" 2>&1 \
                && [ -f "$dir/$d.pt" ] && touch "$dir/$d.done"
        ) &
        pids+=($!)
    done
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done
echo "[rep] training finished: $(ls "$OUT"/run*/ckpt/*.done 2>/dev/null | wc -l) controllers done"

# ---- Eval: each benchmark with its domain's controller.
pids=()
for r in $RUNS; do
    for b in $BENCHES; do
        d=$(domain_of "$b"); out=$OUT/run$r/${b}_qmix.json
        if has_results "$out"; then echo "[rep] run$r $b: done"; continue; fi
        if [ ! -f "$OUT/run$r/ckpt/$d.done" ]; then echo "[rep] run$r $b: no $d controller, skipping"; continue; fi
        limit=(); [ -n "$EVAL_LIMIT" ] && limit=(--limit "$EVAL_LIMIT")
        $PY -m experiments.run_qmix_eval --dataset "$b" --split "$(split_of "$b")" --llm_name "$MODEL" \
            --num_rounds "$(rounds_of "$d")" --model_path "$OUT/run$r/ckpt/$d.pt" --mode qmix \
            "${limit[@]}" --output_path "$out" > "$OUT/run$r/$b.log" 2>&1 &
        pids+=($!)
    done
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

python3 "$REPO/experiments/summarize_replication.py" "$OUT_ROOT"
