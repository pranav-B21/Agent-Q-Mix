# RUNS — Agent-Q-Mix / GCSL Setting B on TACC Vista

One entry per SLURM job or interactive run. Each entry gives the hostname, the absolute paths
of its outputs, what it ran and what came out. Chronological narrative is in `JOURNAL.md`.

All paths are on TACC Vista (`vista.tacc.utexas.edu`), allocation `ASC26032`, partition `gh`
(1× NVIDIA GH200 96 GB per node, aarch64). `$SCRATCH` = `/scratch/11138/pranavbelligundu`.
**`$SCRATCH` purges unaccessed files.** Copy anything that must survive.

Code: `/work/11138/pranavbelligundu/vista/gcsl_mas/Agent-Q-Mix`, a clone of
`github.com/ericjiang18/Agent-Q-Mix` at upstream `c115476`, plus uncommitted local changes
(listed in `JOURNAL.md`, 2026-10-26). There is no fork remote yet.

---

## Shared reference: serving and environments

| | |
|---|---|
| LLM | `Qwen/Qwen3-1.7B`, thinking disabled (`chat_template_kwargs.enable_thinking=False`), Q-Mix default temperature 0.2 |
| Server | vLLM 0.32.0 from `gcsl_mas/.venv_vllm` (torch 2.13.0+cu130), `--max-model-len 16384`, `VLLM_USE_FLASHINFER_SAMPLER=0` (required: no ninja/nvcc) |
| Q-Mix env | `gcsl_mas/.venv_qmix`: torch 2.14.1+cu130, torch-geometric 2.8.1, `setuptools<80` (class-registry needs `pkg_resources`) |
| Endpoint | `.env`: `BASE_URL=http://localhost:8234/v1`, `API_KEY=EMPTY` (an exported `BASE_URL` overrides it) |
| Agents | math: MathSolver ×2 + AnalyzeAgent; MMLU-Pro: ReasoningAgent ×2 + AnalyzeAgent; decision node `FinalRefer` |
| Caches | `HF_HOME=$SCRATCH/hf_home` (Qwen3-1.7B, MMLU-Pro, MATH-500, AIME 2024) |

The `.venv_vllm` and the :8234 server are **shared with Setting A** (`gcsl_mas/setting_a/`).
Don't change their torch/vllm pins, and don't restart the server while Setting A is using it.

---

## 2026-10-26 — vLLM server, interactive

- **Host:** `c642-031.vista.tacc.utexas.edu`, idev job `1102859`.
- **Command:** `vllm serve Qwen/Qwen3-1.7B --port 8234 --served-model-name Qwen/Qwen3-1.7B --gpu-memory-utilization 0.5 --max-model-len 16384`
- **Log:** `/work/11138/pranavbelligundu/vista/gcsl_mas/logs/vllm_qwen3_1.7b.log`
- **Result:** The first attempt died with `FileNotFoundError: 'ninja'` from FlashInfer's sampler
  JIT. The relaunch with `VLLM_USE_FLASHINFER_SAMPLER=0` came up, and "17*3" returned 51.

## 2026-10-26 — Q-Mix smoke train, interactive

- **Host:** `c642-031`, idev job `1102859`.
- **Command:** `python -m experiments.run_qmix_train --dataset mmlu_pro,aime_2024 --split test --llm_name Qwen/Qwen3-1.7B --num_episodes 4 --num_rounds 2 --batch_size 2 --data_limit 2 --token_penalty 0.10 --accuracy_weight 1.50 --device cpu --save_path ../logs/smoke_qmix.pt`
- **Outputs:**
  - Log: `/work/11138/pranavbelligundu/vista/gcsl_mas/logs/smoke_qmix_train.log`
  - Checkpoint: `/work/11138/pranavbelligundu/vista/gcsl_mas/logs/smoke_qmix.pt`
- **Result:** Exit 0. Loss went 0.0148 → 0.0041. Accuracy was 0 on all 4 episodes (the MMLU-Pro
  misses are real wrong answers).

| | MMLU-Pro | AIME 2024 |
|---|---|---|
| Time per episode | 3.8–22.1 s | 47–56 s |
| Tokens per episode | 1,824–5,592 | 6,951–10,933 |

## 2026-10-26 — Generator test, interactive

- **Host:** `c642-031`.
- **Command:** `gen_offline_data --datasets mmlu_pro:2,math500_int:2 --repeats 2`, run as 2 shards.
- **Output:** `/work/11138/pranavbelligundu/vista/gcsl_mas/logs/gen_test/shard_{0,1}.{jsonl,log}`
- **Result:** 8 episodes, 4 correct. Same-task token spread was 1.7k vs 7.9k (one MATH problem)
  and 1.9k vs 5.3k (one MMLU question, both correct). A rerun of shard 0 skipped all 4 finished
  jobs, and truncating a partial last line was checked by hand.

## 2026-10-26 → 10-27 — Offline dataset `offline_v0`, interactive (in place of sbatch)

- **Host:** `c642-031`, idev job `1102859`. It ran against the shared :8234 server (gpu-mem 0.5),
  24 shards as `nohup` processes, started ~23:59 CDT.
- **Command (per shard s = 0..23):** `BASE_URL=http://localhost:8234/v1 python -W ignore -m experiments.gen_offline_data --datasets mmlu_pro:300,math500_int:311 --repeats 3 --seed 0 --shard_id s --num_shards 24 --out $SCRATCH/gcsl_mas/offline_v0/shard_XX.jsonl`
- **Output:** `/scratch/11138/pranavbelligundu/gcsl_mas/offline_v0/shard_{00..23}.{jsonl,log}` (~14 MB)
- **Design:** uniform-random behavior policy (ε = 1); 611 tasks × 3 repeats = 1,833 episodes;
  2 rounds; per-episode topology seed = crc32(`seed|dataset|task_id|repeat`).
- **Throughput:** ~1.3 episodes/s with 24 concurrent shards on one GH200.
- **Result:** **Complete.** 1,833/1,833 episodes, 0 errors, finished ~00:20 CDT 2026-10-27,
  inside the allocation. No sbatch job was needed.

  | | math500_int | mmlu_pro |
  |---|---|---|
  | Episodes | 933 | 900 |
  | Accuracy | 0.690 | 0.467 |
  | Tokens p10 / p50 / p90 | 1,136 / 2,162 / 4,284 | 1,562 / 3,726 / 7,822 |
  | Tasks with mixed outcomes across 3 repeats | 108/311 (34.7%) | 108/300 (36.0%) |
  | Always right / always wrong | 157 / 46 | 90 / 102 |
  | Within-task token spread (max/min), median / p90 | 1.93× / 3.50× | 1.92× / 4.44× |
  | Accuracy range by action used | 0.666 (debate) – 0.711 (agg_refine) | 0.456 (sel_query) – 0.485 (agg_refine) |

  `debate` has the lowest median tokens in both domains (1,975 and 3,096).
- **Regenerate or extend:** submit from a login node with
  `cd /work/11138/pranavbelligundu/vista/gcsl_mas/Agent-Q-Mix && sbatch scripts/sbatch_gen_offline.sh`.
  The job's log goes to `$SCRATCH/gcsl_mas/logs/gen_offline_<jobid>.log`. It skips episodes
  already present, so it only fills gaps in `offline_v0`.
- **Summarize:** `../.venv_qmix/bin/python -m experiments.summarize_offline_data $SCRATCH/gcsl_mas/offline_v0`

## 2026-10-28 — Re-verification on a fresh node, interactive

- **Host:** `c642-011.vista.tacc.utexas.edu`, idev job `1106734`. The GPU was shared with the
  Setting A session, so each of us messages the other before using it.
- **sbatch script, run locally at tiny scale:** `PORT=8240 OUT_DIR=$SCRATCH/gcsl_mas/test_1028_1431 DATASETS="mmlu_pro:2,math500_int:2" REPEATS=2 NUM_SHARDS=2 bash scripts/sbatch_gen_offline.sh`
  - Exit 0. vLLM was ready after 82 s.
  - 8/8 episodes, 0 errors. The end-of-run summary printed.
  - Output: `/scratch/11138/pranavbelligundu/gcsl_mas/test_1028_1431/`
- **QMIX training smoke test, now including `math500_int`:** `run_qmix_train --dataset mmlu_pro,math500_int --num_episodes 4 --data_limit 2 ...`, against vLLM on :8240.
  - Exit 0. Accuracy 1, 1, 0, 0. Loss went 3.65 → 1.28 → 0.65.
  - Log: `/work/11138/pranavbelligundu/vista/gcsl_mas/logs/smoke2_qmix_train.log`
  - Checkpoint: `/work/11138/pranavbelligundu/vista/gcsl_mas/logs/smoke2_qmix.pt`
- **Cleanup lesson.** A `cmd & VP=$!` chain that starts with `cd ... &&` backgrounds a
  subshell, so `kill $VP` killed the subshell and left vLLM running with 49 GB. `pkill -f
  <pattern>` also kills the calling shell when the pattern appears in that shell's own command
  line. Kill vLLM by its own PID and confirm with `nvidia-smi` that memory drops back to ~0.

## 2026-10-28 — Offline QMIX v0, interactive

- **Host:** `c642-011.vista.tacc.utexas.edu`, idev job `1106734`.
- **Training (CPU, 1 core):** `python -m experiments.train_offline_qmix --data_dir $SCRATCH/gcsl_mas/offline_v0 --out_dir $SCRATCH/gcsl_mas/offline_qmix_v0 --train_steps 2000`
  - Defaults: batch 32, lr 5e-4, γ 0.99, target update every 200 steps, w_acc 1.5, w_token 0.1,
    max_tokens 10k, hold-out 15% of tasks (seed 1).
  - Data: 1,833 episodes → 1,557 train. 92 held-out tasks, with 276 behavior episodes.
    Mean train-episode reward 0.8376.
  - Speed: ~0.16 s/step. Loss 0.61 at step 100, 0.52 at step 200.
  - Outputs in `/scratch/11138/pranavbelligundu/gcsl_mas/offline_qmix_v0/`: `qmix_offline.pt`,
    `heldout_tasks.json`, `train_log.json`, `train.log`
- **Eval:** greedy (ε = 0) on the 92 held-out tasks, with vLLM on :8240 at 0.85 GPU memory.
  - Server PID: `gcsl_mas/logs/vllm_8240.pid`. Server log: `gcsl_mas/logs/vllm_8240.log`.
  - Command: `gen_offline_data --checkpoint .../qmix_offline.pt --epsilon 0 --policy_tag offline_qmix_greedy --task_list .../heldout_tasks.json --repeats 2`
  - Output: `$SCRATCH/gcsl_mas/offline_qmix_v0/eval/`
  - Compare: `python -m experiments.compare_policies --behavior_dir .../offline_v0 --policy_dir .../offline_qmix_v0/eval --task_list .../heldout_tasks.json`
- **Training result:** **Q-values diverged.** Q_tot went 1.39 (step 100) → 2.37 (step 400) →
  12.8 (step 1,700). The largest real return is about 3.0 (1.5 per step × 2 steps). Loss bottomed
  out at 0.43 (step 400), then rose to ~1.1. The likely cause is the untargeted last step
  (JOURNAL 2026-10-28).
- **Eval result:** 184 greedy episodes (92 tasks × 2), 0 errors. Paired within task against
  the 276 random-topology behavior episodes; bootstrap 95% CI over tasks:

  | | behavior | offline QMIX | diff | 95% CI |
  |---|---|---|---|---|
  | MATH accuracy (47 tasks) | 0.660 | 0.734 | +0.074 | [+0.000, +0.149] |
  | MATH tokens | 2,785 | 4,099 | +1,314 | [+895, +1,777] |
  | MMLU-Pro accuracy (45 tasks) | 0.496 | 0.489 | −0.007 | [−0.107, +0.085] |
  | MMLU-Pro tokens | 4,299 | 5,453 | +1,153 | [+542, +1,730] |
  | All: accuracy | 0.580 | 0.614 | +0.034 | [−0.029, +0.094] |
  | All: tokens | 3,526 | 4,761 | +1,236 | [+865, +1,598] |
  | All: reward (w_acc 1.5, w_tok 0.1) | 0.835 | 0.875 | +0.040 | [−0.054, +0.130] |

  The greedy policy collapses to a few actions: solo 55%, broadcast 19%, exec_verify 12%. In
  round 0, agent 0 mostly broadcasts (74%) and agent 1 mostly goes solo (81%).
- **Reading:** No significant reward gain over random topologies. It spends ~35% more tokens
  (significant) for a borderline accuracy gain on MATH only. That is consistent with the
  diverged Q. The w_token of 0.1 per 10k tokens is also too small to penalize +1.2k tokens.

## 2026-10-28 — Offline QMIX v0 with terminal target, plus fixed-topology baselines, interactive

- **Host:** `c642-011`, idev job `1106734`. vLLM on :8240 (Qwen3-1.7B, 0.85 GPU memory),
  stopped by PID afterwards.
- **Corrected training:** `train_offline_qmix ... --terminal_target --out_dir $SCRATCH/gcsl_mas/offline_qmix_v0_tt`
  - Same data, split and hyperparameters as v0.
  - Q_tot stayed bounded: 0.93 (step 200) → 1.20 (step 600) → 1.37 (step 1,900). Loss ~0.47–0.52.
  - So the divergence in v0 came from the untargeted last step.
- **Evals:** all on the same 92 held-out tasks × 2 repeats = 184 episodes each, 0 errors.
  - Fixed-topology mode: `gen_offline_data --fixed_action A`, where every agent takes action A
    every round.
  - Outputs: `$SCRATCH/gcsl_mas/offline_qmix_v0_tt/eval/` and
    `$SCRATCH/gcsl_mas/fixed_baselines/{solo,broadcast,agg_refine,debate}/`
- **Results:** each policy minus the random-topology behavior policy, paired within task,
  bootstrap 95% CI over the 92 tasks. Behavior: accuracy 0.580, 3,526 tokens, reward 0.835.

  | Policy | Accuracy | Tokens | Reward | Reward diff [95% CI] |
  |---|---|---|---|---|
  | **fixed all-agg_refine** | **0.674** | 4,144 | **0.970** | **+0.134 [+0.065, +0.206]** |
  | fixed all-"debate"\* | 0.609 | **1,070** | 0.902 | +0.067 [−0.061, +0.199] |
  | fixed all-solo | 0.641 | 9,099 | 0.885 | +0.049 [−0.033, +0.131] |
  | offline QMIX v0 (upstream loss) | 0.614 | 4,761 | 0.875 | +0.040 [−0.054, +0.130] |
  | random topology (behavior) | 0.580 | 3,526 | 0.835 | — |
  | fixed all-broadcast | 0.511 | 3,043 | 0.736 | −0.100 [−0.206, +0.006] |
  | offline QMIX v0 + terminal target | 0.489 | 3,063 | 0.703 | −0.132 [−0.249, −0.020] |

  Per domain, all-agg_refine: MATH 0.787 (+0.128 [+0.064, +0.195]), MMLU-Pro 0.556
  (+0.059 [−0.004, +0.126]).

  \* **"debate" is broken.** With all agents on action 5, no agent node executes. Only the
  FinalRefer decision node runs, answering alone in one LLM call (traced: 210 tokens; under
  all-solo, all 3 agents plus the decision node run, 1,958 tokens). So this row is really a
  single-model baseline. The cause is in `graph.py` `apply_qmix_actions` / `_execute_round`: the
  mutual debate edges leave no agent with in-degree 0. Not yet fixed. Episodes that mix debate
  with other actions may also skip agents.
- **Reading:**
  1. Neither offline QMIX beats the best constant topology (all-agg_refine). The terminal-target
     version is significantly worse than random. Its Q is bounded, but its greedy policy is bad.
  2. One call to the decision model (the "debate" row) matches random multi-agent topologies
     on accuracy at 30% of the tokens. Any controller has to beat that.
  3. This matches the paper's released results, where the trained controllers pick one constant
     topology on 98–100% of items (see JOURNAL).

## 2026-10-28 — gpt-oss-120b load test and a 5-item paper-eval check, interactive

- **Host:** `c642-011`, idev job `1106734`.
- **Weights:** `openai/gpt-oss-120b`, downloaded with `hf download ... --exclude "metal/*"
  --exclude "original/*"`. 15 files, 61 GB, in `$SCRATCH/hf_home`. (`du` on Lustre under-reported
  the folder as 109 KB; `du -L` on the snapshot shows the true size.)
- **Serve:** `vllm serve openai/gpt-oss-120b --port 8241 --gpu-memory-utilization 0.90 --max-model-len 32768`
  with `VLLM_USE_FLASHINFER_SAMPLER=0`.
  - Ready after 200 s.
  - It answered "sum of divisors of 36" with `content="91"` and a separate reasoning field
    (63 reasoning tokens).
  - **vLLM 0.32 serves gpt-oss on aarch64 GH200.**
  - Log: `gcsl_mas/logs/vllm_gptoss_test.log`. Stopped by PID afterwards.
- **Paper eval check:** `run_qmix_eval --dataset mmlu_pro --split validation --llm_name openai/gpt-oss-120b --num_rounds 2 --model_path checkpoints/qmix_unified.pt --mode qmix --limit 5`
  - Output: `$SCRATCH/gcsl_mas/paper_repro_test/`
  - The checkpoint loads. 5/5 correct, same as the paper's file on those items.
  - **But the actions differ.** The same released checkpoint chose `[1,2,2]` (4 items) and
    `[0,2,2]`, while `result/result_gpt/mmlu_pro_qmix.jsonl` shows `[0,0,0]` on all five.
    Tokens were 1.2–1.7k per item here against 3.5–5.6k in the paper's file.
  - Eval is greedy (ε = 0), and the actions depend only on task text, agent IDs and the
    checkpoint. The one later code change (`c115476`, 2026-04-09, after the 03-28 results) only
    changes node execution order. So the released checkpoint very likely isn't the one that
    produced the released results.
- **Status:** the full job, `scripts/sbatch_paper_repro.sh`, is ready to submit, with the
  weights already cached.

## 2026-10-28 — Paper reproduction, sbatch job 1107264 (submitted, pending)

- **Submitted** from `login1` as `sbatch scripts/sbatch_paper_repro.sh`, with the defaults:
  gpt-oss-120b, the released `checkpoints/qmix_unified.pt`, and benchmarks
  `mmlu_pro humaneval aime_2025`.
- **Log:** `/scratch/11138/pranavbelligundu/gcsl_mas/logs/paper_repro_1107264.log`
- **Outputs:** `/scratch/11138/pranavbelligundu/gcsl_mas/paper_repro_gpt_oss/{bench}_qmix.{json,jsonl}`,
  plus per-benchmark logs and the vLLM log.
- **Ran:** 2026-10-30 06:37–07:04 on `c637-021`, 26 min, exit 0. vLLM ready after 220 s.
- **CORRECTION 2026-10-31:** the "Paper" column below is the repo's `result/result_gpt/` files,
  **not the paper**. Those files don't match the paper's Table 1 (`gcsl_mas/Q-MIX.pdf`; median of
  3 runs). So the "Not reproduced" verdict for AIME is withdrawn; see the Table 1 comparison in
  the next entry.
- **Result (gpt-oss-120b, released `checkpoints/qmix_unified.pt`):**

  | Benchmark | Paper | Repro | Item-level | Verdict |
  |---|---|---|---|---|
  | MMLU-Pro (n=70) | 0.900 | 0.943 | both right 63, paper-only 0, repro-only 3, both wrong 4; McNemar p=0.25 | No significant difference |
  | AIME 2025 (n=30) | 0.800 | 0.533 | both right 15, paper-only 9, repro-only 1, both wrong 5; McNemar p=0.021 | Not reproduced |
  | HumanEval | 0.866 | **did not run** | 0 samples; the 0.000 in the log table is not a score | Open |

- **HumanEval failure:** `datasets/humaneval_dataset.py` loads `openai_humaneval`, which the
  current `datasets` library rejects (`HfUriError`: repo id must be `namespace/name`). The loader
  swallows the exception and returns 0 samples. `openai/openai_humaneval` loads all 164 (tested).
  **Trap:** the job skips any benchmark whose JSON exists, and the empty
  `humaneval_qmix.json` exists, so a plain resubmit skips HumanEval. Delete that file or make the
  skip check require `total > 0`.
- **Checkpoint test (CPU, recomputed greedy actions, no LLM):**

  | Checkpoint (step) | MMLU-Pro match to published actions | AIME 2025 match |
  |---|---|---|
  | `checkpoints/` GPT-OSS (143) | 0/70 | 0/30 |
  | `checkpoints_gemini/` (43), control | 66/70 | 30/30 |

  The code reproduces Gemini's published actions, so the released GPT checkpoint is **not** the
  one behind the published GPT results. Published: `(0,0,0)` on 100/100. This checkpoint:
  `(1,2,2)` on 97/100 recomputed. The job is therefore a different topology from the published
  one, and paper-vs-repro differences confound topology with everything else.
- **Tokens are not comparable.** `llm/price.py` estimates `len(text)//4` on prompt and visible
  completion, and reasoning is not counted when content is non-empty. For MMLU + AIME, Q-Mix
  reported 281,567 "tokens". vLLM's throughput log gives ~150k prompt + ~329k generated ≈ 479k
  real tokens, about 1.7× more (estimate from 10 s throughput samples). **Retracts the earlier
  "about a third of the tokens" reading** of the 5-item check.
- **AIME gap breakdown (gap = 8 items: paper 24, repro 16):**
  - 8 final answers begin like raw reasoning ("We need to evaluate…"): repro 0/8, paper 4/8.
  - The other 22: repro 16, paper 20.
  - So these answers account for 4 of the 8, and ordinary wrong answers for the other 4.
  - Hypothesis, not measured: the decision node hits the 4096 `max_tokens` while still reasoning,
    content comes back empty, and `llm/gpt_chat.py` falls back to the reasoning text, which is
    then graded on its last number. `finish_reason` is not logged. Detection was by prefix match.
- **Saved `prediction` fields are cut at 500 chars** by `run_qmix_eval.py`; grading uses the full
  text. Don't judge truncation from them.
- **Outputs:** `/scratch/11138/pranavbelligundu/gcsl_mas/paper_repro_gpt_oss/`

## 2026-10-31 — Repro against the paper's Table 1, and the execution bug

- **Host:** `c642-091`, idev job `1112423`. CPU only, stubbed LLM (`tests/trace_topologies.py`,
  `tests/test_episode_pipeline.py`).
- **Job 1107264 against Table 1 (GPT-OSS:120B):** exact binomial, treating the table as fixed.

  | Benchmark | Table 1 | Repo files | Repro | p vs Table 1 |
  |---|---|---|---|---|
  | MMLU-Pro | 92.86 | 90.00 | 94.29 (66/70) | 0.82 |
  | AIME 2025 | 63.33 | 80.00 | 53.33 (16/30) | 0.26 |
  | HumanEval | 97.56 | 86.59 | not run | — |

  Both are consistent with the table, but this is weak confirmation: n is small, it's one run
  not a median of 3, and the setup differs. The paper uses 3 per-domain checkpoints, 50 episodes,
  w_acc 1.25 for GPT-OSS. `run_all_gpt.sh` and the released checkpoint use one unified checkpoint,
  150 episodes, w_acc 1.50. Other Table 1 vs repo files: AIME26 60.00/80.00, HMMT 53.33/60.00,
  Beyond 42.00/50.00, LiveCodeBench 100/100.
- **Execution bug in `_execute_round` (since first commit `611a33c`):** `self.nodes` is keyed
  by names (`MathSolver_0`) but lookups use random node ids, so they always fail. Any agent with an
  incoming edge never runs and no agent sees a peer's answer. Traced with a stub:

  | Action (all agents) | Agents run: before | after | Peers seen: before | after |
  |---|---|---|---|---|
  | solo | 3/3 | 3/3 | 0 | 0 |
  | broadcast, sel_query, agg_refine, exec_verify | 1/3 | 3/3 | 0 | 2 |
  | debate | 0/3 | 3/3 | 0 | 2 |

  This supersedes the 10-28 explanation of debate (mutual edges). It was not specific to debate.
  **Every Q-Mix result so far ran with this bug:** `offline_v0`, fixed baselines, offline QMIX, job
  1107264. So "all-agg_refine" really meant the AnalyzeAgent alone, then the decision node;
  "all-broadcast" meant one MathSolver alone. Those were comparisons of which agents ran, not of
  communication patterns. Whether the paper's own runs had the bug is unknown: the public repo has
  it from its first commit, and we can't see what code produced Table 1.
- **Step 1 fixes (uncommitted):**
  - `graph/graph.py`: dependency-ordered execution; `run_round` and `decide`; `QMIX_LEGACY_EXEC=1`
    keeps the old behaviour (`sbatch_paper_repro.sh` sets it).
  - `run_episode`: the decision node runs once per episode, and round 2 now sees round 1.
  - `llm/price.py`, `llm/gpt_chat.py`: server-reported tokens including reasoning;
    `finish_reason` and reasoning fallback recorded per call; `LLM_MIN_MAX_TOKENS` sets the cap.
  - `gen_offline_data.py`: schema 2 logs the graph that ran, agents run, per-agent tokens,
    truncation.
  - `utils/task_encoder.py`, `QMIX_TASK_ENCODER=st`: MiniLM projected to 32 dims (similar pair
    0.858, unrelated pair −0.053). Default stays `hash`, so old checkpoints keep working.
  - `sbatch_gen_offline.sh`: defaults to `offline_v1` with the encoder on.
- **Known gap:** `run_qmix_eval.py` still picks actions once and reuses them every round, while
  training picks per round. Not changed, so the paper's eval stays as published.

## 2026-10-31 — Step 1 real-model smoke test, interactive

- **Host:** `c642-091`, Qwen3-1.7B on :8240, 2 shards, `QMIX_TASK_ENCODER=st`.
- **Output:** `/scratch/11138/pranavbelligundu/gcsl_mas/step1_smoke/`
- **Result:** 4 math episodes recorded with schema 2 fields and server-reported tokens
  (`tokens_estimated=False`, 3.3k–19.9k). 3 of 4 ran all 3 agents in both rounds (7 LLM calls).
  One logged 2 truncated calls, so truncation logging works.
- **Not a clean pass:** 4 MMLU episodes failed with `ClientConnectorError`, and one math episode
  ran 1/3 agents in round 1. The wait loop ran out at 180 s, so these are most likely from starting
  before vLLM was up, not confirmed. **Repeat this with a proper ready check before regenerating.**
- **Open:** an agent that fails all retries doesn't fail the episode. It is only visible through
  `executed_agents`, so filter on that or make the episode error out.

## 2026-11-02 — Replication job tests, interactive

- **Host:** `c642-032`, idev job `1116331`. gpt-oss-120b on :8234. Output:
  `/scratch/11138/pranavbelligundu/gcsl_mas/replicate_test/`
- **Scripts:** `scripts/sbatch_replicate_qmix.sh` (VARIANT=released|fixed, paper recipe),
  `scripts/sbatch_replicate_baselines.sh`, `experiments/run_base_eval.py`,
  `experiments/summarize_replication.py`.
- **Released and fixed:** each trained 3 controllers (2 episodes) and scored 4 benchmarks (2 items),
  0 errors. Calls per item: released 1–4; fixed 7 (2 rounds) or 10 (3 rounds), all 3 agents ran.
  Fixed LiveCodeBench: 5 truncated calls and 1 reasoning fallback on one item.
- **Baselines:** Base ran one item. All 4 frameworks first crashed on `pkg_resources`
  (`.venv_base` now has `setuptools<80`). Agent Framework then needed `agent-framework-openai`,
  now installed. Lobster, AutoGen and LangGraph started, but those attempts hit connection errors
  because the server had stopped. No framework is confirmed to get a scored answer yet.
