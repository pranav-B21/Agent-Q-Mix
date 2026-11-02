"""
Compare a policy's eval episodes against the behavior policy on the same held-out tasks.

Pairs within task: for each held-out task, the policy's mean outcome minus the behavior
policy's mean outcome over its logged repeats. Reports means and a bootstrap 95% CI over
tasks, per dataset, plus the policy's action distribution.

Usage:
  python -m experiments.compare_policies --behavior_dir $SCRATCH/gcsl_mas/offline_v0 \
      --policy_dir $SCRATCH/gcsl_mas/offline_qmix_v0/eval \
      --task_list $SCRATCH/gcsl_mas/offline_qmix_v0/heldout_tasks.json
"""

import json
import argparse
import collections
import numpy as np

from experiments.summarize_offline_data import load, ACTION_NAMES


def reward(r, w_acc, w_tok, max_tokens):
    return r["accuracy"] * w_acc - min(r["total_tokens"] / max_tokens, 1.0) * w_tok


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behavior_dir", required=True)
    p.add_argument("--policy_dir", required=True)
    p.add_argument("--task_list", required=True)
    p.add_argument("--accuracy_weight", type=float, default=1.50)
    p.add_argument("--token_penalty", type=float, default=0.10)
    p.add_argument("--max_tokens", type=int, default=10000)
    args = p.parse_args()

    with open(args.task_list) as f:
        held = {tuple(k) for k in json.load(f)}
    beh = collections.defaultdict(list)
    for r in load(args.behavior_dir):
        if (r["dataset"], r["task_id"]) in held:
            beh[(r["dataset"], r["task_id"])].append(r)
    pol = collections.defaultdict(list)
    for r in load(args.policy_dir):
        pol[(r["dataset"], r["task_id"])].append(r)

    keys = sorted(k for k in pol if k in beh)
    print(f"held-out tasks: {len(held)} | with policy episodes: {len(keys)} | "
          f"policy episodes: {sum(len(pol[k]) for k in keys)} | behavior episodes: {sum(len(beh[k]) for k in keys)}\n")

    metrics = {
        "accuracy": lambda r: r["accuracy"],
        "tokens": lambda r: r["total_tokens"],
        "reward": lambda r: reward(r, args.accuracy_weight, args.token_penalty, args.max_tokens),
    }
    for ds in sorted({k[0] for k in keys}) + ["ALL"]:
        ks = [k for k in keys if ds == "ALL" or k[0] == ds]
        print(f"== {ds} ({len(ks)} tasks)")
        print(f"   {'metric':<9} {'behavior':>10} {'policy':>10} {'diff':>10}   95% CI (paired over tasks)")
        for name, fn in metrics.items():
            b = np.array([np.mean([fn(r) for r in beh[k]]) for k in ks])
            q = np.array([np.mean([fn(r) for r in pol[k]]) for k in ks])
            lo, hi = boot_ci(q - b)
            print(f"   {name:<9} {b.mean():>10.3f} {q.mean():>10.3f} {(q - b).mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]")

    acts = collections.Counter()
    per_round = collections.defaultdict(collections.Counter)
    for k in keys:
        for r in pol[k]:
            for t, s in enumerate(r["steps"]):
                for i, a in enumerate(s["actions"]):
                    acts[a] += 1
                    per_round[(t, i)][a] += 1
    total = sum(acts.values())
    print("\npolicy action share: " + ", ".join(f"{ACTION_NAMES[a]} {c / total:.0%}" for a, c in acts.most_common()))
    for (t, i), c in sorted(per_round.items()):
        top, n = c.most_common(1)[0]
        print(f"   round {t} agent {i}: most common {ACTION_NAMES[top]} ({n / sum(c.values()):.0%})")


if __name__ == "__main__":
    main()
