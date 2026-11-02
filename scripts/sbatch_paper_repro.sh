#!/bin/bash
#SBATCH -J qmix-paper-repro
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/gcsl_mas/logs/paper_repro_%j.log
# ============================================================
#  Reproduce Agent Q-Mix's published GPT-OSS-120B numbers (result/result_gpt/) with the
#  released checkpoint (checkpoints/qmix_unified.pt), serving openai/gpt-oss-120b locally.
#  Eval commands are copied from scripts/run_all_gpt.sh; the three benchmarks run as
#  parallel processes against one vLLM server. Ends with a paper-vs-repro table.
#
#  Idempotent per benchmark: a benchmark whose output JSON exists is skipped.
#  Submit from a login node, from the Agent-Q-Mix repo root:
#    sbatch scripts/sbatch_paper_repro.sh
#    BENCHES="mmlu_pro" sbatch scripts/sbatch_paper_repro.sh       # subset
# ============================================================
set -uo pipefail

GCSL=/work/11138/pranavbelligundu/vista/gcsl_mas
REPO=$GCSL/Agent-Q-Mix
MODEL=${MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8234}
CKPT=${CKPT:-$REPO/checkpoints/qmix_unified.pt}
BENCHES=${BENCHES:-"mmlu_pro humaneval aime_2025"}
OUT_DIR=${OUT_DIR:-/scratch/11138/pranavbelligundu/gcsl_mas/paper_repro_gpt_oss}

export HF_HOME=/scratch/11138/pranavbelligundu/hf_home
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer's sampler JIT needs ninja+nvcc, absent here
export PYTHONPATH=$REPO
export QMIX_LEGACY_EXEC=1              # the execution the published results used (graph.py)
export BASE_URL=http://localhost:$PORT/v1 API_KEY=EMPTY

mkdir -p "$OUT_DIR" /scratch/11138/pranavbelligundu/gcsl_mas/logs
cd "$REPO"
echo "[repro] host=$(hostname) job=${SLURM_JOB_ID:-none} model=$MODEL ckpt=$CKPT benches=$BENCHES out=$OUT_DIR"

VLOG="$OUT_DIR/vllm_${SLURM_JOB_ID:-local}.log"
"$GCSL/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" --served-model-name "$MODEL" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 360); do
    curl -sf "localhost:$PORT/v1/models" >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "[repro] vLLM died at startup; see $VLOG"; tail -20 "$VLOG"; exit 1; }
    sleep 5
done
curl -sf "localhost:$PORT/v1/models" >/dev/null || { echo "[repro] vLLM not ready after 30 min"; exit 1; }
echo "[repro] vLLM ready after $((i * 5))s"

# Per-benchmark settings from scripts/run_all_gpt.sh
bench_args() {
    case $1 in
        mmlu_pro)  echo "--split validation --num_rounds 2" ;;
        humaneval) echo "--num_rounds 2" ;;
        aime_2025) echo "--split train --num_rounds 3" ;;
        aime_2026) echo "--split train --num_rounds 3" ;;
        hmmt_2025) echo "--split train --num_rounds 3" ;;
        beyond_aime) echo "--split test --num_rounds 3" ;;
        *) echo "[repro] unknown bench $1" >&2; return 1 ;;
    esac
}

pids=()
for b in $BENCHES; do
    out="$OUT_DIR/${b}_qmix.json"
    if [ -f "$out" ]; then echo "[repro] $b: $out exists, skipping"; continue; fi
    args=$(bench_args "$b") || continue
    echo "[repro] $b: starting ($args)"
    "$GCSL/.venv_qmix/bin/python" -W ignore -m experiments.run_qmix_eval \
        --dataset "$b" $args --llm_name "$MODEL" --model_path "$CKPT" \
        --mode qmix --output_path "$out" > "$OUT_DIR/${b}.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[repro] paper (result/result_gpt) vs this run:"
"$GCSL/.venv_qmix/bin/python" - "$OUT_DIR" "$REPO/result/result_gpt" <<'EOF'
import json, os, sys, glob, collections
out, paper = sys.argv[1], sys.argv[2]
name = {"mmlu_pro": "mmlu_pro", "humaneval": "humaneval", "aime_2025": "aime2025",
        "aime_2026": "aime2026", "hmmt_2025": "hmmt2025", "beyond_aime": "beyond_aime"}
print(f"{'bench':<12} {'paper acc':>10} {'repro acc':>10} {'paper tok':>10} {'repro tok':>10}  repro actions")
for f in sorted(glob.glob(f"{out}/*_qmix.json")):
    b = os.path.basename(f)[: -len("_qmix.json")]
    r = json.load(open(f))["metrics"]
    pf = f"{paper}/{name.get(b, b)}_qmix.json"
    p = json.load(open(pf))["metrics"] if os.path.exists(pf) else {}
    acts = collections.Counter()
    jl = f[:-5] + ".jsonl"
    if os.path.exists(jl):
        for line in open(jl):
            acts.update(json.loads(line).get("actions", []))
    tot = sum(acts.values()) or 1
    print(f"{b:<12} {p.get('accuracy', float('nan')):>10.3f} {r['accuracy']:>10.3f} "
          f"{p.get('avg_tokens', float('nan')):>10.0f} {r['avg_tokens']:>10.0f}  "
          + ", ".join(f"{a}:{c / tot:.0%}" for a, c in acts.most_common(3))
          + f"  (n={r['total']})")
EOF
