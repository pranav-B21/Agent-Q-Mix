"""
Trace which agents run under each constant topology, with a stubbed LLM (no GPU, no server).

For each action: the edges built, the agents that executed in order, and whether each agent's
prompt contained another agent's answer from the current round.

Usage: PYTHONPATH=. python tests/trace_topologies.py
"""

import asyncio
import torch
import llm.gpt_chat as gc
from graph.graph import QMIXGraph

ACTIONS = ["solo", "broadcast", "sel_query", "agg_refine", "exec_verify", "debate"]
AGENTS = ["MathSolver", "MathSolver", "AnalyzeAgent"]


def trace(action):
    calls = []

    async def fake_achat(model_name, messages, max_tokens=1024, temperature=0.2):
        user = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
        calls.append(user)
        return f"stub answer {len(calls)}. The answer is 7."

    gc.achat = fake_achat
    g = QMIXGraph(domain="aime", llm_name="Qwen/Qwen3-1.7B", agent_names=AGENTS, decision_method="FinalRefer")
    g.apply_qmix_actions(torch.full((3,), action, dtype=torch.long))
    adj = g.get_adj_matrix().astype(int).tolist()
    asyncio.run(g.arun({"task": "What is 3+4?"}, num_rounds=1, actions=torch.full((3,), action, dtype=torch.long)))
    n_agent_calls = len(calls) - 1  # the last call is the decision node
    saw_peer = sum("stub answer" in c for c in calls[:-1])
    return adj, n_agent_calls, saw_peer


if __name__ == "__main__":
    print(f"{'action':<12} {'agents run':>10} {'saw a peer':>10}  edges")
    for a, name in enumerate(ACTIONS):
        adj, ran, saw = trace(a)
        print(f"{name:<12} {ran:>7}/3 {saw:>10}  {adj}")
