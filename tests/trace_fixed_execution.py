"""
Check who runs, in what order, and what each agent actually receives (stubbed LLM, no GPU, no server).

Every stub answer is tagged with its agent and round, so a prompt shows exactly whose answers it
contains. Expected edges come from a separate implementation of the action rules, not graph.py.

Sections:
  1. Readable trace of the six all-same-action topologies over 2 rounds.
  2. Edges against the paper's action definitions, before graph.py's cycle guard drops any.
  3. All 216 joint actions through run_episode (training) and arun (eval).
  4. An agent that fails every retry.

Usage:
  PYTHONPATH=. python tests/trace_fixed_execution.py
  QMIX_LEGACY_EXEC=1 PYTHONPATH=. python tests/trace_fixed_execution.py   # must report failures
"""

import asyncio
import itertools
import logging
import random
import re
import sys

import torch

import graph.graph as graph_module
import llm.gpt_chat as gc
from experiments.run_qmix_train import get_obs_dim, get_state_dim, run_episode
from graph.graph import QMIXGraph
from llm.price import cost_count
from qmix.agent_network import NUM_ACTIONS
from qmix.qmix_trainer import QMIXTrainer

ACTIONS = ["solo", "broadcast", "sel_query", "agg_refine", "exec_verify", "debate"]
AGENTS = ["MathSolver", "MathSolver", "AnalyzeAgent"]
N = len(AGENTS)
USAGE = {"prompt_tokens": 100, "completion_tokens": 50}
TOKENS_PER_CALL = 150
TAG = re.compile(r"ANS_a(\d)_r(\d)")


class Recorder:
    """Instruments one graph: which agent called the LLM, in which round, with what prompt."""

    def __init__(self, g, fail_agent=None, fail_round=None):
        self.calls = []  # (agent index or "D", round, set of (agent, round) tags seen)
        self.agent = None
        self.round = 0
        self.fail_agent, self.fail_round = fail_agent, fail_round
        for i, nid in enumerate(g.node_ids):
            self._wrap_node(g.nodes[nid], i)
        self._wrap_node(g.decision_node, "D")
        run_round = g.run_round

        async def tracked_round(input, actions, round_idx, *args, **kwargs):
            self.round = round_idx
            await run_round(input, actions, round_idx, *args, **kwargs)

        g.run_round = tracked_round

    def _wrap_node(self, node, who):
        execute = node.async_execute

        async def tracked(*args, **kwargs):
            self.agent = who
            return await execute(*args, **kwargs)

        node.async_execute = tracked

    async def achat(self, model_name, messages, max_tokens=1024, temperature=0.2):
        if self.agent == self.fail_agent and self.round == self.fail_round:
            raise RuntimeError("stub failure")
        text = "\n".join(m["content"] if isinstance(m, dict) else m.content for m in messages)
        seen = {(int(a), int(r)) for a, r in TAG.findall(text)}
        self.calls.append((self.agent, self.round, seen))
        cost_count(text, "x", model_name, usage=USAGE, finish_reason="stop", reasoning_fallback=False)
        if self.agent == "D":
            return "Decision. The answer is 7."
        return f"ANS_a{self.agent}_r{self.round}. The answer is 7."


def reference_edges(actions, guard=True):
    """Action rules from the paper (sec. 4), reimplemented. guard=False skips the cycle check."""
    edges = set()

    def reaches(a, b):
        stack, seen = [a], set()
        while stack:
            x = stack.pop()
            if x == b:
                return True
            if x not in seen:
                seen.add(x)
                stack += [v for u, v in edges if u == x]
        return False

    def add(src, dst, mutual=False):
        if guard and reaches(dst, src):
            return
        edges.add((src, dst))
        if mutual:
            edges.add((dst, src))

    for i, a in enumerate(actions):
        if a == 1:
            for j in range(N):
                if j != i:
                    add(i, j)
        elif a == 2:
            add(i, (i + 1) % N)
        elif a == 3:
            for j in range(N):
                if j != i:
                    add(j, i)
        elif a == 4 and i + 1 < N:
            add(i, i + 1)
        elif a == 5:
            add(i, (i + N // 2) % N, mutual=True)
    return edges


def expected_inputs(edges, agent, rnd):
    """A predecessor's answer from this round; the first of a mutual pair gets last round's."""
    seen = {(agent, rnd - 1)} if rnd > 0 else set()
    for u, v in edges:
        if v != agent:
            continue
        if (v, u) in edges and agent < u:
            if rnd > 0:
                seen.add((u, rnd - 1))
        else:
            seen.add((u, rnd))
    return seen


def check_round(calls, edges, rnd):
    """Problems in one round's agent calls, compared with its edges."""
    problems = []
    order = [c[0] for c in calls if c[1] == rnd and c[0] != "D"]
    if sorted(order) != list(range(N)):
        problems.append(f"round {rnd}: ran {order}, want each of 0..{N - 1} once")
    for u, v in edges:
        if u in order and v in order:
            first, second = (min(u, v), max(u, v)) if (v, u) in edges else (u, v)
            if order.index(first) > order.index(second):
                problems.append(f"round {rnd}: {second} ran before {first}")
    for who, r, seen in calls:
        if r == rnd and who != "D" and seen != expected_inputs(edges, who, rnd):
            problems.append(f"round {rnd}: agent {who} saw {sorted(seen)}, want {sorted(expected_inputs(edges, who, rnd))}")
    return problems


def check_decision(calls, last_round):
    decisions = [c for c in calls if c[0] == "D"]
    want = {(i, last_round) for i in range(N)}
    if len(decisions) != 1:
        return [f"decision ran {len(decisions)} times"]
    return [] if decisions[0][2] == want else [f"decision saw {sorted(decisions[0][2])}, want {sorted(want)}"]


def make_trainer():
    obs = get_obs_dim(N)
    return QMIXTrainer(n_agents=N, obs_dim=obs, state_dim=get_state_dim(N, obs), n_actions=NUM_ACTIONS)


def train_episode(trainer, per_round, **fail):
    """The real training path, with the policy replaced by a fixed list of actions per round."""
    g = QMIXGraph(domain="aime", llm_name="Qwen/Qwen3-1.7B", agent_names=AGENTS, decision_method="FinalRefer")
    rec = Recorder(g, **fail)
    gc.achat = rec.achat
    script = iter(per_round)
    trainer.select_actions = lambda obs, adj, h, eps: (torch.tensor(next(script), dtype=torch.long), h)
    ep, answers = asyncio.run(run_episode(g, trainer, "What is 3+4?", 1.0, num_rounds=len(per_round)))
    return ep, answers, rec


def edges_of(adj):
    return {(i, j) for i in range(N) for j in range(N) if adj[i][j]}


def check_train(trainer, per_round):
    ep, _, rec = train_episode(trainer, per_round)
    problems = []
    for r, actions in enumerate(per_round):
        want = reference_edges(actions)
        got = edges_of(ep.steps[r].executed_adj)
        if got != want:
            problems.append(f"round {r}: edges {sorted(got)}, want {sorted(want)}")
        problems += check_round(rec.calls, want, r)
    problems += check_decision(rec.calls, len(per_round) - 1)
    want_tokens = (N * len(per_round) + 1) * TOKENS_PER_CALL
    if ep.total_tokens != want_tokens:
        problems.append(f"tokens {ep.total_tokens}, want {want_tokens}")
    return problems


def check_eval(actions, rounds=3):
    g = QMIXGraph(domain="aime", llm_name="Qwen/Qwen3-1.7B", agent_names=AGENTS, decision_method="FinalRefer")
    rec = Recorder(g)
    gc.achat = rec.achat
    asyncio.run(g.arun({"task": "What is 3+4?"}, num_rounds=rounds, actions=torch.tensor(actions, dtype=torch.long)))
    want = reference_edges(actions)
    problems = []
    for r in range(rounds):
        problems += check_round(rec.calls, want, r)
    return problems + check_decision(rec.calls, rounds - 1)


def label(who):
    return "decision" if who == "D" else f"agent {who}"


def show_tags(seen):
    return ", ".join(f"a{a} r{r}" for a, r in sorted(seen, key=lambda t: (t[1], t[0]))) or "only the task"


def section_trace(trainer):
    print("=" * 78 + "\n1. TRACE, same action for all agents, 2 rounds (a1 r0 = agent 1's round-0 answer)\n" + "=" * 78)
    for a, name in enumerate(ACTIONS):
        actions = [a] * N
        ep, _, rec = train_episode(trainer, [actions, actions])
        print(f"\n{name}: edges {sorted(edges_of(ep.steps[0].executed_adj)) or 'none'}")
        for who, r, seen in rec.calls:
            print(f"  round {r}  {label(who):<9} received: {show_tags(seen)}")


def section_paper():
    print("\n" + "=" * 78 + "\n2. EDGES vs the paper's rules, before the cycle guard\n" + "=" * 78)
    for a, name in enumerate(ACTIONS):
        full = reference_edges([a] * N, guard=False)
        kept = reference_edges([a] * N)
        print(f"  {name:<12} rules give {len(full)}, kept {len(kept)}, dropped {sorted(full - kept) or 'none'}")
    joint = list(itertools.product(range(NUM_ACTIONS), repeat=N))
    changed = sum(reference_edges(j, guard=False) != reference_edges(j) for j in joint)
    print(f"  joint actions that lose at least one edge: {changed}/{len(joint)}")


def section_exhaustive(trainer):
    print("\n" + "=" * 78 + "\n3. ALL 216 JOINT ACTIONS\n" + "=" * 78)
    joint = list(itertools.product(range(NUM_ACTIONS), repeat=N))
    rng = random.Random(0)
    warnings = []
    graph_module.logger.warning = lambda msg, *a, **k: warnings.append(str(msg))
    suites = [
        ("training, same actions both rounds", [[list(j), list(j)] for j in joint], lambda c: check_train(trainer, c)),
        ("training, different actions per round", [[list(j), list(rng.choice(joint))] for j in joint for _ in range(5)],
         lambda c: check_train(trainer, c)),
        ("eval path (arun), 3 rounds", [list(j) for j in joint], check_eval),
    ]
    total_bad = 0
    for title, cases, fn in suites:
        failures = [(c, p) for c in cases if (p := fn(c))]
        total_bad += len(failures)
        print(f"  {title}: {len(cases) - len(failures)}/{len(cases)} pass")
        for case, problems in failures[:2]:
            print(f"    e.g. {case}: {problems[0]}")
    print(f"  cycle or retry warnings logged: {len(warnings)}")
    return total_bad


def section_failure(trainer):
    print("\n" + "=" * 78 + "\n4. AGENT 0 FAILS EVERY RETRY IN ROUND 0, broadcast\n" + "=" * 78)
    graph_module.logger.warning = lambda msg, *a, **k: None
    ep, answers, rec = train_episode(trainer, [[1, 1, 1], [1, 1, 1]], fail_agent=0, fail_round=0)
    for who, r, seen in rec.calls:
        print(f"  round {r}  {label(who):<9} received: {show_tags(seen)}")
    print(f"  executed_agents per round: {[s.executed_agents for s in ep.steps]}")
    print(f"  episode raised: no; answer returned: {bool(answers)}")


if __name__ == "__main__":
    logging.disable(logging.INFO)
    print(f"LEGACY_EXECUTION={graph_module.LEGACY_EXECUTION}\n")
    t = make_trainer()
    section_trace(t)
    section_paper()
    bad = section_exhaustive(t)
    section_failure(t)
    print(f"\nRESULT: {'all checks passed' if not bad else f'{bad} cases FAILED'}")
    sys.exit(1 if bad else 0)
