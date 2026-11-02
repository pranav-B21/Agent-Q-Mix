"""
Summarize an offline dataset from gen_offline_data.py: is there enough outcome variation
for hindsight relabeling to carry signal?

Usage:
  python -m experiments.summarize_offline_data /scratch/.../gcsl_mas/offline_v0
"""

import sys
import json
import glob
import collections
import numpy as np

ACTION_NAMES = ["solo", "broadcast", "sel_query", "agg_refine", "exec_verify", "debate"]


def load(out_dir):
    recs = []
    for f in sorted(glob.glob(f"{out_dir}/shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def main(out_dir):
    recs = load(out_dir)
    print(f"{len(recs)} episodes from {out_dir}\n")
    by_ds = collections.defaultdict(list)
    for r in recs:
        by_ds[r["dataset"]].append(r)

    for ds, rs in sorted(by_ds.items()):
        acc = np.array([r["accuracy"] for r in rs])
        tok = np.array([r["total_tokens"] for r in rs])
        print(f"== {ds}: n={len(rs)}  acc={acc.mean():.3f}  "
              f"tokens p10/p50/p90={np.percentile(tok, 10):.0f}/{np.median(tok):.0f}/{np.percentile(tok, 90):.0f}")

        # Per-task variation across repeats (different random topologies, same task).
        tasks = collections.defaultdict(list)
        for r in rs:
            tasks[r["task_id"]].append(r)
        multi = {t: v for t, v in tasks.items() if len(v) > 1}
        if multi:
            mixed = sum(1 for v in multi.values() if len({x["accuracy"] for x in v}) > 1)
            all_right = sum(1 for v in multi.values() if all(x["accuracy"] == 1 for x in v))
            spread = [max(x["total_tokens"] for x in v) / max(min(x["total_tokens"] for x in v), 1)
                      for v in multi.values()]
            print(f"   tasks with >1 repeat: {len(multi)}  | outcome mixed across repeats: {mixed} "
                  f"({mixed / len(multi):.1%})  | always right: {all_right}  "
                  f"| always wrong: {len(multi) - mixed - all_right}")
            print(f"   within-task token spread (max/min): median {np.median(spread):.2f}x, "
                  f"p90 {np.percentile(spread, 90):.2f}x")

        # Marginal effect of each action (counted over all agents and rounds).
        print("   action       share   acc|used  tokens|used")
        for a, name in enumerate(ACTION_NAMES):
            used = [r for r in rs if any(a in s["actions"] for s in r["steps"])]
            share = sum(s["actions"].count(a) for r in rs for s in r["steps"]) / max(
                sum(len(s["actions"]) for r in rs for s in r["steps"]), 1)
            if used:
                print(f"   {name:<12} {share:5.1%}   {np.mean([r['accuracy'] for r in used]):.3f}"
                      f"     {np.median([r['total_tokens'] for r in used]):.0f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
