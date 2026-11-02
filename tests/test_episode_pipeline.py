"""
Episode pipeline checks with a stubbed LLM (no GPU, no server).

Usage: PYTHONPATH=. python tests/test_episode_pipeline.py
"""

import asyncio
import numpy as np
import torch
from qmix.qmix_trainer import QMIXTrainer
from qmix.agent_network import NUM_ACTIONS
from graph.graph import QMIXGraph
from experiments.run_qmix_train import get_obs_dim, get_state_dim, run_episode
from utils.globals import PromptTokens, CompletionTokens, LLMCalls
from llm.price import cost_count
import llm.gpt_chat as gc

AGENTS = ["MathSolver", "MathSolver", "AnalyzeAgent"]
USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 30}}
prompts = []


async def fake_achat(model_name, messages, max_tokens=1024, temperature=0.2):
    prompts.append(messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content)
    cost_count(prompts[-1], "x", model_name, usage=USAGE, finish_reason="stop", reasoning_fallback=False)
    return f"stub {len(prompts)}. The answer is 7."


def episode(action, rounds=2):
    gc.achat = fake_achat
    prompts.clear()
    LLMCalls.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    n = 3
    o = get_obs_dim(n)
    t = QMIXTrainer(n_agents=n, obs_dim=o, state_dim=get_state_dim(n, o), n_actions=NUM_ACTIONS)
    t.select_actions = lambda obs, adj, h, eps: (torch.full((n,), action, dtype=torch.long), h)
    g = QMIXGraph(domain="aime", llm_name="Qwen/Qwen3-1.7B", agent_names=AGENTS, decision_method="FinalRefer")
    ep, answers = asyncio.run(run_episode(g, t, "What is 3+4?", 1.0, num_rounds=rounds))
    return ep, answers


# Every action runs all agents, plus one decision call per episode.
for a in range(NUM_ACTIONS):
    ep, answers = episode(a)
    assert len(prompts) == 3 * 2 + 1, (a, len(prompts))
    assert all(len(s.executed_agents) == 3 for s in ep.steps), a
    assert answers and "answer" in answers[0]
    assert ep.total_tokens == 7 * 150, (a, ep.total_tokens)  # server usage, not chars/4

# Round 2 agents see their own round 1 answer.
episode(0)
assert not any("previously" in p for p in prompts[:3])
assert all("previously" in p for p in prompts[3:6])

# Peers are visible only where the topology has an edge.
episode(0)
assert not any("stub" in p for p in prompts[:3])
ep, _ = episode(3)
assert ep.steps[0].executed_adj.sum() == 3
assert sum("stub" in p for p in prompts[:3]) == 2

# The logged graph is the one that ran; the network input stays the pre-action graph.
assert ep.steps[0].adj_matrix.sum() == 0
assert ep.steps[1].adj_matrix.sum() == 3

# Tokens per agent reach the observation (its token_ratio feature was always 0 before).
assert all(t > 0 for t in ep.steps[0].agent_tokens)
print("all episode pipeline checks passed")
