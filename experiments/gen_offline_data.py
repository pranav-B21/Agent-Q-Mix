"""
Offline dataset generation for goal-conditioned (GCSL) topology controllers.

Runs the same multi-agent episodes as run_qmix_train.py, but under a fixed behavior
policy (uniform-random actions, i.e. epsilon=1) and with no learning, and writes every
episode to JSONL so it can later be relabeled with the outcome it achieved
(accuracy, tokens).

Each task is repeated --repeats times with independently sampled topologies, so the
same task appears with different outcomes. Work is split across processes with
--shard_id/--num_shards; a process resumes by skipping (dataset, task_id, repeat)
keys already present in its --out file.

Usage:
  python -m experiments.gen_offline_data --datasets mmlu_pro:300,math500_int:311 \
      --repeats 3 --shard_id 0 --num_shards 16 --out data/offline/shard_00.jsonl
"""

import os
import sys
import json
import time
import random
import zlib
import argparse
import asyncio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmix.qmix_trainer import QMIXTrainer
from qmix.agent_network import NUM_ACTIONS
from graph.graph import QMIXGraph
from datasets import get_dataset
from utils.globals import LLMCalls
from experiments.run_qmix_train import (
    DOMAIN_MAP, AGENT_CONFIGS, get_obs_dim, get_state_dim, run_episode,
)


def build_jobs(spec: str, repeats: int, seed: int):
    """Expand 'name:count,...' into a deterministic list of (ds_name, dataset, sample, repeat)."""
    jobs = []
    for item in spec.split(","):
        name, _, count = item.strip().partition(":")
        ds = get_dataset(name, split="test")
        if len(ds) == 0:
            raise RuntimeError(f"No samples loaded for {name}")
        # Random subset: HF splits are often grouped (MMLU-Pro by category), so a prefix is biased.
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        idx = idx[: int(count)] if count else idx
        for i in idx:
            for r in range(repeats):
                jobs.append((name, ds, ds[i], r))
    return jobs


def done_keys(path: str):
    keys = set()
    if os.path.exists(path):
        # A killed process can leave an unterminated last line; drop it so appends stay valid JSONL.
        with open(path, "rb+") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                f.truncate(data.rfind(b"\n") + 1)
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line from a killed process
                keys.add((rec["dataset"], rec["task_id"], rec["repeat"]))
    return keys


async def generate(args):
    jobs = build_jobs(args.datasets, args.repeats, args.seed)
    if args.task_list:
        with open(args.task_list) as f:
            keep = {tuple(k) for k in json.load(f)}
        jobs = [j for j in jobs if (j[0], j[2].task_id) in keep]
    jobs = jobs[args.shard_id :: args.num_shards]
    finished = done_keys(args.out)
    todo = [j for j in jobs if (j[0], j[2].task_id, j[3]) not in finished]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(jobs)} jobs, "
          f"{len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

    n_agents = 3
    obs_dim = get_obs_dim(n_agents)
    trainer = QMIXTrainer(
        n_agents=n_agents, obs_dim=obs_dim, state_dim=get_state_dim(n_agents, obs_dim),
        n_actions=NUM_ACTIONS, device="cpu",
    )
    if args.checkpoint:
        trainer.load(args.checkpoint)
    if args.fixed_action is not None:
        # Constant-topology baseline: every agent takes the same action every round.
        fixed = torch.full((n_agents,), args.fixed_action, dtype=torch.long)
        trainer.select_actions = lambda obs, adj, hidden, epsilon: (fixed.clone(), hidden)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fout:
        for k, (ds_name, ds, sample, rep) in enumerate(todo):
            # Per-episode seed so the sampled topology is reproducible regardless of sharding.
            ep_seed = zlib.crc32(f"{args.seed}|{ds_name}|{sample.task_id}|{rep}".encode())
            np.random.seed(ep_seed)

            agent_names = AGENT_CONFIGS[ds_name]
            graph = QMIXGraph(
                domain=DOMAIN_MAP[ds_name], llm_name=args.llm_name,
                agent_names=agent_names, decision_method=args.decision_method,
            )
            t0 = time.time()
            LLMCalls.instance().reset()
            try:
                episode, answers = await run_episode(
                    graph, trainer, sample.task, epsilon=args.epsilon, num_rounds=args.num_rounds,
                )
            except Exception as e:
                print(f"  [error] {ds_name}/{sample.task_id}/r{rep}: {type(e).__name__}: {e}", flush=True)
                continue
            answer = answers[0] if answers else ""
            accuracy = ds.evaluate(answer, sample.ground_truth)

            calls = LLMCalls.instance().value
            rec = {
                "schema": 2,
                "task_encoder": os.getenv("QMIX_TASK_ENCODER", "hash"),
                "llm_calls": len(calls),
                "completion_tokens": sum(c["completion_tokens"] for c in calls),
                "tokens_estimated": any(c["estimated"] for c in calls),
                "truncated_calls": sum(c.get("finish_reason") == "length" for c in calls),
                "reasoning_fallback_calls": sum(bool(c.get("reasoning_fallback")) for c in calls),
                "decision_finish_reason": calls[-1].get("finish_reason") if calls else None,
                "decision_reasoning_fallback": bool(calls[-1].get("reasoning_fallback")) if calls else False,
                "dataset": ds_name,
                "domain": DOMAIN_MAP[ds_name],
                "task_id": sample.task_id,
                "repeat": rep,
                "episode_seed": ep_seed,
                "behavior_policy": args.policy_tag,
                "epsilon": args.epsilon,
                "llm_name": args.llm_name,
                "agent_names": agent_names,
                "num_rounds": args.num_rounds,
                "task": sample.task,
                "ground_truth": sample.ground_truth,
                "metadata": {k_: v for k_, v in sample.metadata.items() if k_ != "options"},
                "answer": answer,
                "accuracy": accuracy,
                "total_tokens": episode.total_tokens,
                "wall_time_s": round(time.time() - t0, 2),
                "steps": [
                    {
                        "actions": s.actions.tolist(),
                        "adj_matrix": s.adj_matrix.tolist(),
                        "observations": np.round(s.observations, 5).tolist(),
                        "global_state": np.round(s.global_state, 5).tolist(),
                        "token_usage": int(s.token_usage),
                        "executed_adj": s.executed_adj.astype(int).tolist(),
                        "executed_agents": s.executed_agents,
                        "agent_tokens": s.agent_tokens,
                    }
                    for s in episode.steps
                ],
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"  [{k+1}/{len(todo)}] {ds_name}/{sample.task_id}/r{rep} "
                  f"acc={accuracy:.0f} tokens={episode.total_tokens} "
                  f"actions={[s.actions.tolist() for s in episode.steps]} {rec['wall_time_s']}s",
                  flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="mmlu_pro:300,math500_int:311",
                   help="comma list of name:count (count omitted = all)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--decision_method", default="FinalRefer")
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=None, help="QMIX checkpoint to act with (default: untrained net)")
    p.add_argument("--epsilon", type=float, default=1.0, help="1.0 = uniform random; 0.0 = greedy")
    p.add_argument("--policy_tag", default="uniform_random", help="label stored in each record")
    p.add_argument("--fixed_action", type=int, default=None,
                   help="constant-topology baseline: all agents take this action (0-5) every round")
    p.add_argument("--task_list", default=None, help="JSON list of [dataset, task_id] to restrict to")
    asyncio.run(generate(p.parse_args()))


if __name__ == "__main__":
    main()
