"""
QMIX-integrated Multi-Agent Graph.

Orchestrates agent execution using communication topologies learned by QMIX.
The graph manages:
  1. Dynamic spatial/temporal connections based on QMIX actions
  2. Topological-order execution of agent nodes
  3. Integration with QMIX for action selection and reward collection

Communication patterns are determined by QMIX agent actions:
  - solo_process: no edges for this agent
  - broadcast_all: edges to all neighbors
  - selective_query: edge to best neighbor
  - aggregate_refine: receive from all, send refined
  - execute_verify: code execution + tool use
  - debate_check: adversarial edges
"""

import os
import shortuuid
import numpy as np
import torch
import asyncio
from typing import Any, List, Optional, Dict, Tuple

from graph.node import Node
from agents.agent_registry import AgentRegistry
from prompt.prompt_set_registry import PromptSetRegistry
from qmix.agent_network import ACTION_NAMES
from utils.log import get_logger
from utils.globals import PromptTokens, CompletionTokens

logger = get_logger("graph")

# QMIX_LEGACY_EXEC=1 keeps upstream's execution, where any agent with an incoming edge never
# runs (see _execute_round_legacy). Only for reproducing the published results.
LEGACY_EXECUTION = os.getenv("QMIX_LEGACY_EXEC", "0") == "1"


TOPOLOGY_PRESETS = {
    "solo": lambda n: np.zeros((n, n)),
    "star": lambda n: _star_adj(n),
    "chain": lambda n: _chain_adj(n),
    "full": lambda n: np.ones((n, n)) - np.eye(n),
    "ring": lambda n: _ring_adj(n),
}


def _star_adj(n):
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = 1
        adj[i, 0] = 1
    return adj


def _chain_adj(n):
    adj = np.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def _ring_adj(n):
    adj = _chain_adj(n)
    if n > 2:
        adj[0, n - 1] = 1
        adj[n - 1, 0] = 1
    return adj


class QMIXGraph:
    """Multi-agent graph with QMIX-driven communication topology.

    Supports both:
      - Fixed topologies (for baseline comparisons)
      - Dynamic QMIX-selected topologies (for learned policies)
    """

    def __init__(
        self,
        domain: str,
        llm_name: str,
        agent_names: List[str],
        decision_method: str = "FinalRefer",
        fixed_topology: str = None,
    ):
        self.id = shortuuid.ShortUUID().random(length=4)
        self.domain = domain
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.nodes: Dict[str, Node] = {}
        self.prompt_set = PromptSetRegistry.get(domain)

        self._init_nodes()

        self.decision_node = AgentRegistry.get(
            decision_method, domain=self.domain, llm_name=self.llm_name
        )

        self.n_agents = len(self.nodes)
        self.node_ids = list(self.nodes.keys())
        self.last_executed: List[str] = []
        self.last_adj = np.zeros((self.n_agents, self.n_agents))

        if fixed_topology and fixed_topology in TOPOLOGY_PRESETS:
            self._fixed_adj = TOPOLOGY_PRESETS[fixed_topology](self.n_agents)
        else:
            self._fixed_adj = None

    def _init_nodes(self):
        for i, agent_name in enumerate(self.agent_names):
            try:
                node = AgentRegistry.get(agent_name, domain=self.domain, llm_name=self.llm_name)
                node_id = f"{agent_name}_{i}"
                self.nodes[node_id] = node
            except Exception as e:
                logger.warning(f"Failed to create agent {agent_name}: {e}")

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return sum(len(n.spatial_successors) for n in self.nodes.values())

    def get_adj_matrix(self) -> np.ndarray:
        """Current adjacency matrix of spatial connections."""
        n = self.n_agents
        matrix = np.zeros((n, n))
        for i, nid1 in enumerate(self.node_ids):
            for j, nid2 in enumerate(self.node_ids):
                if self.nodes[nid2] in self.nodes[nid1].spatial_successors:
                    matrix[i, j] = 1
        return matrix

    def apply_topology(self, adj_matrix: np.ndarray):
        """Apply adjacency matrix as spatial connections."""
        self._clear_spatial()
        n = min(adj_matrix.shape[0], self.n_agents)
        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] > 0.5:
                    src = self.nodes[self.node_ids[i]]
                    dst = self.nodes[self.node_ids[j]]
                    if not self._check_cycle(dst, {src}):
                        src.add_successor(dst, "spatial")

    def apply_qmix_actions(self, actions: torch.Tensor):
        """Convert QMIX agent actions into communication topology.

        Each agent's action determines its communication pattern:
          0 - solo_process:    no edges
          1 - broadcast_all:   edges to all other agents
          2 - selective_query: edge to one neighbor (round-robin)
          3 - aggregate_refine: receive from all (reversed edges)
          4 - execute_verify:  edge to self (tool use, minimal comm)
          5 - debate_check:    edge to specific partner for debate
        """
        self._clear_spatial()
        n = self.n_agents

        for i, action in enumerate(actions):
            action = action.item()
            src = self.nodes[self.node_ids[i]]

            if action == 0:  # solo
                continue
            elif action == 1:  # broadcast
                for j in range(n):
                    if j != i:
                        dst = self.nodes[self.node_ids[j]]
                        if not self._check_cycle(dst, {src}):
                            src.add_successor(dst, "spatial")
            elif action == 2:  # selective query
                best_j = (i + 1) % n
                dst = self.nodes[self.node_ids[best_j]]
                if not self._check_cycle(dst, {src}):
                    src.add_successor(dst, "spatial")
            elif action == 3:  # aggregate (receive from all)
                for j in range(n):
                    if j != i:
                        other = self.nodes[self.node_ids[j]]
                        if not self._check_cycle(src, {other}):
                            other.add_successor(src, "spatial")
            elif action == 4:  # execute_verify (minimal communication)
                if i + 1 < n:
                    dst = self.nodes[self.node_ids[i + 1]]
                    if not self._check_cycle(dst, {src}):
                        src.add_successor(dst, "spatial")
            elif action == 5:  # debate
                partner = (i + n // 2) % n
                dst = self.nodes[self.node_ids[partner]]
                if not self._check_cycle(dst, {src}):
                    src.add_successor(dst, "spatial")
                    dst.add_successor(src, "spatial")

    async def arun(
        self,
        input: Dict[str, str],
        num_rounds: int = 2,
        max_tries: int = 3,
        max_time: int = 300,
        actions: torch.Tensor = None,
    ) -> Tuple[List[Any], int]:
        """Execute the multi-agent graph.

        Args:
            input: {"task": "..."} the task to solve
            num_rounds: number of communication rounds
            actions: QMIX actions to apply; if None, use fixed topology
        Returns:
            (final_answers, total_tokens_used)
        """
        tokens_before = PromptTokens.instance().value + CompletionTokens.instance().value

        for round_idx in range(num_rounds):
            await self.run_round(input, actions, round_idx, max_tries, max_time)

        final_answers = await self.decide(input)

        tokens_after = PromptTokens.instance().value + CompletionTokens.instance().value
        total_tokens = int(tokens_after - tokens_before)

        return final_answers, total_tokens

    async def run_round(self, input, actions, round_idx, max_tries=3, max_time=300):
        """Run one communication round. round_idx > 0 links each agent to its previous output."""
        if actions is not None:
            self.apply_qmix_actions(actions)
        elif self._fixed_adj is not None:
            self.apply_topology(self._fixed_adj)

        if round_idx > 0:
            self._connect_temporal(round_idx)

        self.last_executed = []
        if LEGACY_EXECUTION:
            await self._execute_round_legacy(input, max_tries, max_time)
        else:
            await self._execute_round(input, max_tries, max_time)
        self.last_adj = self.get_adj_matrix()
        self._update_memory()

    async def decide(self, input):
        """Run the decision node over the agents' latest outputs."""
        self._connect_decision_node()
        await self.decision_node.async_execute(input)
        return self.decision_node.outputs or ["No answer produced"]

    async def _execute_round(self, input, max_tries, max_time):
        """Execute all agents in dependency order, so each sees its predecessors' outputs.

        Debate adds mutual edges (A <-> B). The lower-indexed agent runs first and its partner
        sees that output; the first agent sees its partner's output from the previous round.
        """
        nid_of = {node.id: nid for nid, node in self.nodes.items()}
        order = {nid: i for i, nid in enumerate(self.node_ids)}

        waits_on = {}
        for nid, node in self.nodes.items():
            preds = set()
            for pred in node.spatial_predecessors:
                pred_nid = nid_of.get(pred.id)
                if pred_nid is None:
                    continue
                if pred in node.spatial_successors and order[nid] < order[pred_nid]:
                    continue  # mutual edge: the lower-indexed agent goes first
                preds.add(pred_nid)
            waits_on[nid] = preds

        finished = set()
        while len(finished) < len(self.nodes):
            ready = [nid for nid in self.node_ids if nid not in finished and waits_on[nid] <= finished]
            if not ready:
                # Debate's reverse edge skips _check_cycle, so a joint action like [3,3,5] can
                # form a longer cycle. Run the first unfinished agent rather than skip any.
                ready = [nid for nid in self.node_ids if nid not in finished]
                logger.warning(f"Dependency cycle among {ready}; breaking it at {ready[0]}")
            current_id = ready[0]
            node = self.nodes[current_id]
            before = PromptTokens.instance().value + CompletionTokens.instance().value
            for attempt in range(max_tries):
                try:
                    await asyncio.wait_for(node.async_execute(input), timeout=max_time)
                    self.last_executed.append(current_id)
                    break
                except Exception as e:
                    logger.warning(f"Node {current_id} attempt {attempt + 1} failed: {e}")
            else:
                # A partial team must not be recorded as a normal episode.
                raise RuntimeError(f"Agent {current_id} failed all {max_tries} attempts")
            node.token_usage = int(PromptTokens.instance().value + CompletionTokens.instance().value - before)
            finished.add(current_id)

    def _is_mutual_edge(self, nid_a, nid_b):
        """Check if two nodes share a bidirectional (mutual) edge."""
        node_a = self.nodes[nid_a]
        node_b = self.nodes[nid_b]
        return node_b in node_a.spatial_successors and node_a in node_b.spatial_successors

    async def _execute_round_legacy(self, input, max_tries, max_time):
        """Upstream execution, kept for reproducing published results.

        self.nodes is keyed by names like "MathSolver_0" but pred.id / successor.id are random
        node ids, so both lookups below always fail: inbound edges are counted but never
        released, and any agent with an incoming edge never runs. No agent sees a peer's answer.

        Mutual edges (A <-> B) from debate actions are handled by breaking
        symmetry: the lower-ID node's inbound mutual edge is not counted,
        allowing it to enter the queue first. Once it executes, its partner's
        in-degree drops to zero and executes next with access to the first
        node's output.
        """
        in_degree = {}
        for nid, node in self.nodes.items():
            count = 0
            for pred in node.spatial_predecessors:
                if pred.id in self.nodes and self._is_mutual_edge(nid, pred.id):
                    if nid < pred.id:
                        continue
                count += 1
            in_degree[nid] = count

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        executed = set()

        while queue:
            current_id = queue.pop(0)
            if current_id in executed:
                continue

            for attempt in range(max_tries):
                try:
                    await asyncio.wait_for(
                        self.nodes[current_id].async_execute(input),
                        timeout=max_time,
                    )
                    executed.add(current_id)
                    break
                except Exception as e:
                    logger.warning(f"Node {current_id} attempt {attempt + 1} failed: {e}")

            for successor in self.nodes[current_id].spatial_successors:
                if successor.id not in self.nodes:
                    continue
                in_degree[successor.id] = in_degree.get(successor.id, 1) - 1
                if in_degree[successor.id] == 0 and successor.id not in executed:
                    queue.append(successor.id)

    def _clear_spatial(self):
        for node in self.nodes.values():
            node.spatial_predecessors = []
            node.spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def _clear_temporal(self):
        for node in self.nodes.values():
            node.temporal_predecessors = []
            node.temporal_successors = []

    def _connect_temporal(self, round_idx):
        """Connect temporal edges (self-connections from previous round)."""
        self._clear_temporal()
        if round_idx == 0:
            return
        for nid, node in self.nodes.items():
            if node.last_memory["outputs"]:
                node.add_predecessor(node, "temporal")

    def _connect_decision_node(self):
        for node in self.nodes.values():
            node.add_successor(self.decision_node, "spatial")

    def _update_memory(self):
        for node in self.nodes.values():
            node.update_memory()

    def _check_cycle(self, new_node, target_nodes, visited=None):
        if visited is None:
            visited = set()
        if id(new_node) in visited:
            return False
        visited.add(id(new_node))
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self._check_cycle(successor, target_nodes, visited):
                return True
        return False

    def get_observation_features(self, task: str) -> np.ndarray:
        """Create observation features for QMIX from current state.

        Features per agent: [task_hash_features, agent_role_id, round_info,
                             num_neighbors, has_output, token_usage_ratio]
        """
        obs_list = []
        task_features = self._task_to_features(task)

        for i, nid in enumerate(self.node_ids):
            node = self.nodes[nid]
            agent_feature = np.zeros(16)
            agent_feature[i % 16] = 1.0  # agent identity one-hot (up to 16 agents)

            has_output = 1.0 if node.outputs else 0.0
            n_neighbors = len(node.spatial_predecessors) + len(node.spatial_successors)

            node_obs = np.concatenate([
                task_features,
                agent_feature,
                [has_output, n_neighbors / max(self.n_agents, 1), node.token_usage / 10000.0],
            ])
            obs_list.append(node_obs)

        return np.stack(obs_list)

    def _task_to_features(self, task: str, dim: int = 32) -> np.ndarray:
        """Task features: byte hash by default, or a text embedding with QMIX_TASK_ENCODER=st."""
        if os.getenv("QMIX_TASK_ENCODER", "hash") == "st":
            from utils.task_encoder import encode_task
            return encode_task(task, dim)
        features = np.zeros(dim)
        for i, c in enumerate(task.encode()[:dim * 4]):
            features[i % dim] += c / 255.0
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features

    def get_global_state(self, task: str) -> np.ndarray:
        """Global state for the mixing network: concat of all agent observations + graph stats."""
        obs = self.get_observation_features(task)
        adj = self.get_adj_matrix()
        graph_stats = np.array([
            adj.sum(),
            adj.sum() / max(self.n_agents ** 2, 1),
            self.n_agents,
        ])
        return np.concatenate([obs.flatten(), graph_stats])
