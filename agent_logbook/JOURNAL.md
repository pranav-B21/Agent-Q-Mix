# JOURNAL — Agent-Q-Mix / GCSL Setting B on TACC Vista (Pranav)

Dated narrative: what was tried, what was found, what was decided. Per-run details (hosts,
paths, numbers, settings) are in `RUNS.md`.

This clone lives in `/work/11138/pranavbelligundu/vista/gcsl_mas/`, the workspace for the
offline GCSL multi-agent project. That project has two settings, and this repo is **Setting B**:
a goal-conditioned controller over communication topology. **Setting A** (goal-in-prompt SFT of
one shared LLM) lives in `gcsl_mas/setting_a/` and `gcsl_mas/DrMAS/` and keeps its own records.

---

## 2026-10-26 — Why Q-Mix, and the plan

**Framing (from the PI's slides).** The goal is offline goal-conditioned supervised learning
(GCSL) for multi-agent LLM systems, optimizing accuracy, efficiency and safety. Each logged
trajectory gets relabeled with the outcome it achieved, e.g. (correct?, tokens used, safety
violation?). A policy is then trained by supervised learning to reproduce that trajectory when
given that goal. The PI ruled out training a separate LLM per agent. That leaves two settings:
a shared backbone (A, SHARP-style) and an RL controller over topology (B, Q-Mix-style).

**Why B is a clean test.** Agent Q-Mix (arXiv 2604.00344, code
`github.com/ericjiang18/Agent-Q-Mix`) trains a QMIX controller that picks, per agent and per
round, one of six communication actions (solo, broadcast, selective query, aggregate-refine,
execute-verify, debate). Its reward is `accuracy·w_acc − token_ratio·w_token`, which hard-codes
one trade-off between two of the PI's three objectives. A goal-conditioned controller could
serve any trade-off at test time without retraining. The natural baseline is offline QMIX on
the same buffer. The controller itself is tiny; the cost is LLM inference to build the dataset.

**Planned steps:**
1. Run Q-Mix on a local Qwen3-1.7B.
2. Log a behavior-policy dataset.
3. Train a goal-conditioned controller and sweep the token budget at test time.
4. Reread AgentHER/HSL for how they relabel and whether they handle stochastic outcomes.

## 2026-10-26 — Step 1: Q-Mix on a local Qwen3-1.7B (no API)

**Serving.** vLLM 0.32.0 installs from PyPI on aarch64 (torch 2.13 cu130) and sees the GH200.
The first launch died: FlashInfer's sampler JIT-compiles with `ninja`/`nvcc`, and neither
exists here. Fixed with `VLLM_USE_FLASHINFER_SAMPLER=0`, which falls back to vLLM's PyTorch
sampler. The server runs from the shared `gcsl_mas/.venv_vllm`, also used by Setting A.

**Q-Mix env.** A separate `gcsl_mas/.venv_qmix` (torch 2.14, torch-geometric 2.8.1), so Q-Mix's
requirements can't move vLLM's torch. `class-registry` imports `pkg_resources`, which is gone
from setuptools ≥ 80, so the env pins `setuptools<80`.

**Code changes (uncommitted on upstream `c115476`).**
- `llm/gpt_chat.py`: registers `Qwen/Qwen3-1.7B` and sends `enable_thinking=False` for Qwen3,
  matching the repo's own `scripts/serve_qwen.py`.
- `.env` (gitignored) points at `localhost:8234`.

**Smoke train passed** (4 episodes, MMLU-Pro + AIME 2024; see RUNS.md). All four scored 0. The
MMLU misses are real wrong answers, not parsing: the model said H and D where the answers were
I and F. At 1.7B, AIME is ~0, so it gives no relabeling signal.

**Two findings that shape step 3:**
- **Episodes were never persisted.** `run_qmix_train.py` keeps them only in the in-memory
  replay buffer, so there was no offline dataset to reuse.
- **The controller barely sees the task.** `graph.py:_task_to_features` sums the prompt's raw
  bytes into 32 bins. It is not a semantic embedding. A per-task goal-conditioned controller
  will probably need a real text embedding. Raise this with the PI.

## 2026-10-26 → 10-27 — Step 2: offline behavior-policy dataset

**Choice of data.** GCSL needs tasks whose outcomes vary, and AIME at 1.7B does not. The data
uses two domains:
- **MMLU-Pro:** 300 questions, sampled with a fixed seed, because the HF split is grouped by
  category.
- **MATH-500, integer-answer subset:** a new loader, `math500_int`, holding 311 of the 500
  problems across all five levels. It is restricted to integer answers because the repo's math
  grader compares the last number in the text and would misgrade LaTeX answers such as
  `\frac{1}{2}`.

**Behavior policy:** uniform-random actions (ε = 1), no learning. Each task runs 3 times with
independently sampled topologies, so the same task shows up with different outcomes. Total:
611 × 3 = 1,833 episodes. Each topology draw is seeded from (seed, dataset, task, repeat), so
sharding doesn't change it. LLM sampling stays at Q-Mix's temperature 0.2 and is not seeded.

**New code (uncommitted).**
- `experiments/gen_offline_data.py`: reuses `run_episode` from the training script, so episodes
  match what QMIX training sees. Writes one JSONL line per episode with task, ground truth,
  answer, accuracy, total tokens and, per round, actions, adjacency, observations, global state
  and tokens. Shards with `--shard_id/--num_shards` and resumes by skipping finished
  (dataset, task, repeat) keys. A killed process can leave an unterminated last line, so resume
  truncates it first; otherwise the next append would corrupt a second record.
- `experiments/summarize_offline_data.py`: tells whether the data carries relabeling signal.
  Reports the share of tasks with mixed outcomes across repeats, the within-task token spread,
  and each action's marginal effect.
- `scripts/sbatch_gen_offline.sh`: one node running its own vLLM server plus 24 shards. It is
  idempotent and writes to the same output directory.

**How it actually ran.** The idev allocation had ~30 min left, and automated ssh to a login
node needs TACC's password + 2FA, so the sbatch job couldn't be submitted from the session. The
24 shards were started directly on the idev node instead, against the existing :8234 server.
Throughput was ~1.3 episodes/s, enough to cover the whole dataset inside the allocation. The
sbatch script remains the way to regenerate or extend the dataset.

**First read (351 episodes, MMLU-Pro only):**
- Accuracy is 0.47, and 36.5% of tasks have mixed outcomes across their 3 topologies.
- Within a task, the costliest topology uses a median 1.95× the tokens of the cheapest
  (p90 5.25×).
- Each action's marginal effect is small (accuracy 0.45–0.52 by action), so if topology
  matters, it matters per task, which is what a conditional controller has to learn.

**Final (1,833 episodes, 0 errors; table in RUNS.md).** The pattern holds in both domains:
- About a third of tasks (34.7% MATH, 36.0% MMLU-Pro) have mixed outcomes across topologies.
  Those are the tasks where relabeling can teach "this topology hits this goal".
- Within a task, token cost varies about 1.9× at the median.
- Each action's marginal effect on accuracy stays within ±3 points.

Caveat: with 3 repeats and temperature 0.2, part of the outcome variation is LLM sampling
noise rather than topology. Separating the two (re-running a fixed topology) is the
"You Can't Count on Luck" check, and should come before claiming a goal-conditioned
controller learned anything.

**Open questions for the PI (Wednesday 2026-10-28):**
- Does a generated behavior-policy dataset count as "offline"?
- There is no safety signal yet.
- Stochastic outcomes: a lucky run gets relabeled as reliable (Paster et al., "You Can't Count
  on Luck").
- The byte-hash task features.

## 2026-10-28 — Re-verification, then the first offline QMIX run

**Re-verification on c642-011.** The current code was retested on a fresh idev node, which
shares its GPU with the Setting A session. Both sessions now send a one-line message before
using it. The tiny local run of `sbatch_gen_offline.sh` and a 4-episode QMIX training on
MMLU-Pro + MATH-500 both passed (RUNS.md). The other two GCSL jobs: `gcsl-gen-offline`
(1103387) completed as a no-op, because the data was already there. `gcsl-setA` (1103384) failed
at startup because the vLLM log name contained the slash in `Qwen/Qwen3-1.7B`. The Setting A
session fixed it; I tested the fix live and it was resubmitted as 1107162.

**Offline QMIX, the baseline for the GCSL controller.** The unchanged `QMIXTrainer` is trained
purely from `offline_v0`, with no environment interaction.
- **Rewards** are recomputed as in `run_qmix_train.py`, with the `run_all_*.sh` weights:
  w_acc 1.5, w_token 0.1, max_tokens 10k. As upstream does, the episode reward is put on every
  step.
- **Held-out split:** 15% of *tasks* per dataset (92 tasks, split seed 1). Every repeat of a
  held-out task is excluded, leaving 1,557 training episodes.
- **Free baseline:** the 276 logged random-topology episodes on those same held-out tasks.
- **Eval:** greedy actions (ε = 0) on the held-out tasks, paired against that baseline within
  task.

**New code.**
- `experiments/train_offline_qmix.py`
- `experiments/compare_policies.py`: paired within task, bootstrap CI, action distribution
- `gen_offline_data.py` gained `--checkpoint`, `--epsilon`, `--policy_tag` and `--task_list`, so
  the same script both generates data and evaluates a policy.

**Caveat about upstream's loss.** In `train_step`,
`targets = rewards[:, :-1] + γ·(1−done)·Q_tot_target[:, 1:]`, so the last step never gets its
own target. With 2 rounds, only the round-1 Q-values are fit directly. The round-2 Q-values
enter only through the target network's max. This is kept as is so the baseline is
upstream-faithful, but round-2 greedy actions may be close to arbitrary. Check the per-round
action distribution in the eval.

Results: see RUNS.md, "Offline QMIX v0".

## 2026-10-28 (afternoon) — Is the paper usable? Fixed-topology baselines and the released results

**The paper's released results** (`result/result_{gpt,gemini}/*.jsonl` store each item's actions):
- The GPT-OSS-120B controller picks all-solo `(0,0,0)` on 100% of items in every benchmark
  except LiveCodeBench.
- The Gemini controller picks all-aggregate-refine `(3,3,3)` on 98–100% of items.
- So outside LiveCodeBench, the reported numbers come from a **constant topology**, not from
  per-task topology selection.
- **LiveCodeBench 400/400 (1.0) for both models is a grading artifact.**
  `LiveCodeBenchDataset.evaluate` (`datasets/livecodebench_dataset.py:53`) scores 1 whenever the
  answer is longer than 10 characters, without running any code.

**Fixed-topology baselines on our data** (table in RUNS.md):
- The best constant topology (all-agg_refine) beats random topologies by +0.134 reward and +9.4
  points of accuracy. Both offline-QMIX variants fall short of it.
- A single call to the decision model ("debate", which runs no agents because of a Q-Mix bug)
  matches random topologies' accuracy at 30% of the tokens.
- What this means for GCSL Setting B: the claim to test is not "beats random topologies" but
  "beats the best fixed topology at each token budget". The comparison set should be the
  constant topologies plus single-model, and the controller has to win somewhere on that
  accuracy-vs-token frontier.

**Implications for using Q-Mix:**
- Its environment code (agents, graph, datasets) works locally and is a usable testbed.
- Before trusting results, fix the debate action, which deadlocks agents.
- Replace the LiveCodeBench grader if code tasks are used.
- Its headline claim of learned topology selection is not supported by its own released
  outputs.

**Option A, reproducing the paper's numbers:** `scripts/sbatch_paper_repro.sh` serves
`openai/gpt-oss-120b` (~65 GB MXFP4) with vLLM and runs the paper's own eval commands
(`run_all_gpt.sh`) with the released `checkpoints/qmix_unified.pt` on MMLU-Pro (paper 0.900),
HumanEval (0.866) and AIME 2025 (0.800). It ends with a paper-vs-repro table.
- **Not yet run.** The weights download (`hf download openai/gpt-oss-120b --exclude "metal/*"
  --exclude "original/*"`) was started on the idev node but won't finish there. Run it on a
  login node first.
- **Unverified:** that vLLM 0.32 serves gpt-oss on aarch64 GH200.

**Update (same afternoon).** gpt-oss-120b downloaded fully (61 GB) and serves on the GH200
under vLLM 0.32. A 5-item run of the paper's own MMLU-Pro eval with the released checkpoint got
5/5, as the paper did. But it picked different topologies (`[1,2,2]`, not the paper's
`[0,0,0]`) at about a third of the tokens. With deterministic greedy selection, that means the
released checkpoint does not reproduce the released runs. The full repro job will show whether
the *accuracy* numbers hold anyway. If they do, it supports the reading that topology barely
matters for a strong model on these benchmarks.

## 2026-10-31 — Paper reproduction results (job 1107264)

**MMLU-Pro holds (0.943 vs 0.900, p=0.25). AIME 2025 does not (0.533 vs 0.800, p=0.021).
HumanEval never ran** because of a dataset-name failure, so its 0.000 is not a result. Details in
RUNS.md.

**Confirmed: the released GPT-OSS checkpoint did not produce the published GPT-OSS results.**
This replaces the argument of 10-28 with a test: recomputed actions match published on 0/100
items, while the Gemini checkpoint, as a control, matches on 96/100. So this job cannot confirm
the published GPT-OSS numbers as an exact rerun. It tests a different topology with the same
model. The near-constant-topology finding survives: this checkpoint picks `(1,2,2)` on 97% of
items.

**Correction to 10-28.** I reported that the repro used "about a third of the tokens" and
suggested topology might explain it. Q-Mix's token count is chars/4 of visible text and leaves out
gpt-oss reasoning, so that ratio doesn't measure cost. Real usage in this job was about 1.7× what
Q-Mix reported. `offline_v0` uses the same estimator (with Qwen thinking off, so nothing is hidden):
its numbers are fine for comparing within the dataset, but they are estimates, not real token
counts.

**Decisive next test:** run all-solo `(0,0,0)` with gpt-oss on AIME 2025, which matches the
published topology. If accuracy returns to ~0.8, the AIME gap is topology. If not, it is the
serving setup (for example the 4096 cap with reasoning fallback), or the published number doesn't
reproduce. Also log `finish_reason`, and rerun HumanEval with the fixed dataset name.

## 2026-10-31 (later) — Two corrections

**The AIME verdict above was wrong.** I compared against the repo's result files, assuming they
were the paper's numbers. The paper's Table 1 differs from them, and against it the repro is within
noise (AIME 2025 53.33 vs 63.33, p=0.26). The Setting A session caught this. The result files'
provenance is unclear, so what was said from them (constant topology, "wrong checkpoint") describes
the released artifacts, not necessarily the runs behind Table 1. Table 1 itself is unconfirmed:
one run, small n, different training recipe, and HumanEval missing.

**The bigger finding: agents never communicated.** See RUNS.md. Fixed, with the old behaviour
behind a flag. All earlier topology comparisons need redoing. `offline_v0` is retired.
