"""
Compare replication runs with the paper's Table 1 (GPT-OSS:120B rows, accuracy %).

Reads <root>/{released,fixed}/run*/<bench>_qmix.json and <root>/baselines/*.json. Q-Mix cells are
the median across runs, as in the paper, with the number of runs in brackets.

Usage: python experiments/summarize_replication.py /scratch/.../gcsl_mas/replicate_qmix
"""

import os
import sys
import glob
import json
import statistics

BENCHES = ["livecodebench", "humaneval", "mmlu_pro", "aime_2025", "aime_2026", "hmmt_2025", "beyond_aime"]
TABLE1 = {
    "Agent Q-Mix": [100.00, 97.56, 92.86, 63.33, 60.00, 53.33, 42.00],
    "AutoGen": [100.00, 95.12, 81.43, 53.33, 70.00, 43.33, 37.00],
    "LangGraph": [93.50, 93.90, 45.86, 46.67, 60.00, 30.00, 25.00],
    "Agent Framework": [98.25, 95.73, 80.00, 53.33, 63.33, 30.00, 32.00],
    "Lobster": [96.00, 96.34, 87.14, 63.33, 60.00, 6.67, 15.00],
    "Base": [88.75, 90.04, 74.29, 23.33, 20.00, 26.67, 18.00],
}
FILES = {"AutoGen": "autogen", "LangGraph": "langgraph", "Agent Framework": "agent-framework", "Lobster": "lobster"}


def row(name, cells):
    print(f"{name:<24}" + "".join(f"{c:>12}" for c in cells))


def accuracy(path):
    d = json.load(open(path))
    m = d.get("metrics", d)
    total = m.get("total", m.get("total_samples", 0))
    if not total:
        return None
    return 100 * m.get("accuracy", 0)


def qmix_cells(root, variant):
    cells, tokens = [], []
    for b in BENCHES:
        vals, toks = [], []
        for f in sorted(glob.glob(f"{root}/{variant}/run*/{b}_qmix.json")):
            acc = accuracy(f)
            if acc is not None:
                vals.append(acc)
                toks.append(json.load(open(f))["metrics"]["total_tokens"])
        cells.append(f"{statistics.median(vals):.2f}({len(vals)})" if vals else "-")
        tokens.append(f"{statistics.median(toks) / 1000:.0f}K" if toks else "-")
    return cells, tokens


def main(root):
    print("\nAccuracy (%). Q-Mix cells: median(number of runs).")
    print("LiveCodeBench uses the repo's grader, which only checks answer length.\n")
    row("", [b.replace("livecodebench", "livecode").replace("_", "") for b in BENCHES])
    row("Table 1: Agent Q-Mix", [f"{v:.2f}" for v in TABLE1["Agent Q-Mix"]])
    released, rel_tok = qmix_cells(root, "released")
    fixed, fix_tok = qmix_cells(root, "fixed")
    row("Q-Mix, released code", released)
    row("Q-Mix, fixed code", fixed)
    print()
    for name, values in TABLE1.items():
        if name == "Agent Q-Mix":
            continue
        row(f"Table 1: {name}", [f"{v:.2f}" for v in values])
        stem = FILES.get(name, "base")
        cells = []
        for b in BENCHES:
            p = f"{root}/baselines/{stem}_{b}.json"
            acc = accuracy(p) if os.path.exists(p) else None
            cells.append(f"{acc:.2f}" if acc is not None else "-")
        row(f"  ours", cells)
    print("\nTotal tokens, server-reported, median across runs:")
    row("Q-Mix, released code", rel_tok)
    row("Q-Mix, fixed code", fix_tok)


if __name__ == "__main__":
    main(sys.argv[1])
