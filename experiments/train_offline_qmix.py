"""
Offline QMIX: train the unchanged QMIXTrainer on a logged behavior-policy dataset
(gen_offline_data.py output) with no environment interaction.

This is the baseline the goal-conditioned controller is compared against. Rewards are
recomputed exactly as run_qmix_train.py does (accuracy*w_acc - min(tokens/max_tokens,1)*w_token,
assigned to every step of the episode), and train_step() is upstream's TD loss.

A fraction of *tasks* (not episodes) per dataset is held out, so every repeat of a held-out
task is excluded from training. The held-out list is written next to the checkpoint for
gen_offline_data.py --task_list to evaluate on.

Usage:
  python -m experiments.train_offline_qmix --data_dir $SCRATCH/gcsl_mas/offline_v0 \
      --out_dir $SCRATCH/gcsl_mas/offline_qmix_v0 --train_steps 2000
"""

import os
import sys
import json
import time
import random
import argparse
import collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmix.qmix_trainer import QMIXTrainer
from qmix.replay_buffer import Episode, EpisodeStep
from qmix.agent_network import NUM_ACTIONS
from experiments.run_qmix_train import get_obs_dim, get_state_dim
from experiments.summarize_offline_data import load


def split_tasks(recs, holdout_frac, seed):
    """Per dataset, hold out a seeded fraction of task_ids. Returns the held-out key set."""
    by_ds = collections.defaultdict(set)
    for r in recs:
        by_ds[r["dataset"]].add(r["task_id"])
    held = set()
    for ds, ids in sorted(by_ds.items()):
        ids = sorted(ids)
        random.Random(seed).shuffle(ids)
        held |= {(ds, t) for t in ids[: int(round(holdout_frac * len(ids)))]}
    return held


def to_episode(rec, trainer, max_tokens, terminal_target=False):
    reward = trainer.compute_reward(rec["accuracy"], rec["total_tokens"], max_tokens=max_tokens)
    ep = Episode()
    for i, s in enumerate(rec["steps"]):
        ep.add_step(EpisodeStep(
            observations=np.array(s["observations"], dtype=np.float32),
            actions=np.array(s["actions"], dtype=np.int64),
            rewards=np.zeros(len(s["actions"])),
            team_reward=reward,  # as in run_qmix_train.py: episode reward on every step
            adj_matrix=np.array(s["adj_matrix"], dtype=np.float32),
            global_state=np.array(s["global_state"], dtype=np.float32),
            done=(i == len(rec["steps"]) - 1),
            token_usage=s["token_usage"],
        ))
    if terminal_target:
        # Upstream's loss only targets steps [:-1], so the real last step never gets a target and
        # its Q-values drift; round 1 then bootstraps off their max. An absorbing dummy step makes
        # the real last step a target-bearing step (target = r, since done=1 stops bootstrapping);
        # the dummy itself is never a target.
        last = ep.steps[-1]
        ep.add_step(EpisodeStep(
            observations=last.observations, actions=np.zeros_like(last.actions),
            rewards=np.zeros_like(last.rewards), team_reward=0.0, adj_matrix=last.adj_matrix,
            global_state=last.global_state, done=True, token_usage=0,
        ))
    return ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--holdout_frac", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target_update", type=int, default=200)
    # run_all_gpt.sh / run_all_gemini.sh weights
    p.add_argument("--token_penalty", type=float, default=0.10)
    p.add_argument("--accuracy_weight", type=float, default=1.50)
    p.add_argument("--max_tokens", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--terminal_target", action="store_true",
                   help="also fit the last step (target = r); upstream's loss leaves it untargeted")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    recs = load(args.data_dir)
    held = split_tasks(recs, args.holdout_frac, args.split_seed)
    train = [r for r in recs if (r["dataset"], r["task_id"]) not in held]
    print(f"{len(recs)} episodes | train {len(train)} | held-out tasks {len(held)} "
          f"({len(recs) - len(train)} episodes)", flush=True)

    n_agents = len(train[0]["agent_names"])
    obs_dim = get_obs_dim(n_agents)
    trainer = QMIXTrainer(
        n_agents=n_agents, obs_dim=obs_dim, state_dim=get_state_dim(n_agents, obs_dim),
        n_actions=NUM_ACTIONS, lr=args.lr, gamma=args.gamma,
        target_update_interval=args.target_update, buffer_capacity=len(train),
        batch_size=args.batch_size, token_penalty_weight=args.token_penalty,
        accuracy_reward_weight=args.accuracy_weight, device="cpu",
    )
    for r in train:
        trainer.replay_buffer.push(to_episode(r, trainer, args.max_tokens, args.terminal_target))
    rewards = [trainer.compute_reward(r["accuracy"], r["total_tokens"], args.max_tokens) for r in train]
    print(f"buffer: {len(trainer.replay_buffer)} episodes, mean episode reward {np.mean(rewards):.4f}",
          flush=True)

    log = []
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        info = trainer.train_step()
        if step % args.log_interval == 0 or step == args.train_steps:
            info["elapsed_s"] = round(time.time() - t0, 1)
            log.append(info)
            print(f"step {step}: loss={info['loss']:.4f} q_tot={info['q_tot_mean']:.3f} "
                  f"|td|={info['td_error_abs']:.3f} ({info['elapsed_s']}s)", flush=True)

    ckpt = os.path.join(args.out_dir, "qmix_offline.pt")
    trainer.save(ckpt)
    with open(os.path.join(args.out_dir, "heldout_tasks.json"), "w") as f:
        json.dump(sorted([list(k) for k in held]), f)
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "n_train_episodes": len(train), "log": log}, f, indent=1)
    print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
