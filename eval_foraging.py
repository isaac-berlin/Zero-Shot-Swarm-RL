import os
import csv
import numpy as np
import torch
from torch.distributions import Categorical
from itertools import combinations
import imageio.v2 as imageio   # <-- for video recording

from foraging import Foraging
from train_foraging import ActorMLP, ActorCNN


# ============================================================
# Setup folders
# ============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir("videos")


# ============================================================
# Utilities
# ============================================================

def load_actor(path, obs_spec, n_actions, obs_mode, device="cpu"):
    if obs_mode in ("knn", "vector"):
        model = ActorMLP(obs_spec, n_actions)
    else:
        model = ActorCNN(obs_spec, n_actions)

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def policy_step(actor, obs, device="cpu"):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits = actor(obs_t)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return int(action.item())


def mean_pairwise_distance(locs):
    if len(locs) < 2:
        return 0
    dists = [abs(x1 - x2) + abs(y1 - y2) for (x1, y1), (x2, y2) in combinations(locs, 2)]
    return float(np.mean(dists))


# ============================================================
# Run ONE episode (with optional video recording)
# ============================================================

def run_episode(env, actor, device="cpu", max_steps=4000, video_writer=None):
    obs, _ = env.reset()
    agent_order = env.possible_agents[:]

    step = 0
    total_reward = 0
    pairwise_history = []
    done_flags = {a: False for a in agent_order}

    total_possible_reward = env.n_items

    # write initial frame
    if video_writer is not None:
        frame = env.render(mode="rgb_array")
        video_writer.append_data(frame)

    while step < max_steps and not all(done_flags.values()):
        actions = {a: policy_step(actor, obs[a], device) for a in agent_order}

        next_obs, rewards, dones, truncs, infos = env.step(actions)
        total_reward += sum(rewards.values())

        locs = [env.agent_location[a] for a in agent_order]
        pairwise_history.append(mean_pairwise_distance(locs))

        # Save frame
        if video_writer is not None:
            frame = env.render(mode="rgb_array")
            video_writer.append_data(frame)

        obs = next_obs
        done_flags = {a: dones[a] or truncs[a] for a in agent_order}
        step += 1

    items_collected = sum(env.collected[item] for item in env.items)
    completion = 1 if items_collected == env.n_items else 0
    mpd = float(np.mean(pairwise_history)) if pairwise_history else 0

    return completion, step, total_reward / total_possible_reward, mpd


# ============================================================
# Run N episodes and average metrics
# ============================================================

def run_n_episodes(env, actor, n=100, device="cpu"):
    completions, timesteps, rewards, mpds = [], [], [], []

    for _ in range(n):
        c, t, r, m = run_episode(env, actor, device, video_writer=None)
        completions.append(c)
        timesteps.append(t)
        rewards.append(r)
        mpds.append(m)

    return {
        "completion_mean": float(np.mean(completions)),
        "completion_std": float(np.std(completions)),
        "timesteps_mean": float(np.mean(timesteps)),
        "timesteps_std": float(np.std(timesteps)),
        "reward_ratio_mean": float(np.mean(rewards)),
        "reward_ratio_std": float(np.std(rewards)),
        "mpd_mean": float(np.mean(mpds)),
        "mpd_std": float(np.std(mpds)),
    }


# ============================================================
# Test Scenarios
# ============================================================

TEST_CASES = [
    (10, 8, 8),
    (10, 8, 4),
    (10, 4, 8),
    (20, 8, 8),
    (20, 8, 4),
    (20, 4, 8),
]


# ============================================================
# Evaluation Loop (100 episodes + 1 recorded episode)
# ============================================================

def evaluate_policy(obs_mode, actor_path, csv_path, device="cpu"):
    results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "grid", "agents", "items",
            "completion_mean", "completion_std",
            "timesteps_mean", "timesteps_std",
            "reward_ratio_mean", "reward_ratio_std",
            "mpd_mean", "mpd_std"
        ])

        for (grid, nagents, nitems) in TEST_CASES:

            # Build environment
            env = Foraging(
                grid_size=grid,
                num_agents=nagents,
                num_items=nitems,
                obs_mode=obs_mode
            )

            # Determine obs spec for actor
            obs_sample, _ = env.reset()
            agent0 = env.possible_agents[0]
            obs_spec = (
                obs_sample[agent0].shape if obs_mode == "window"
                else obs_sample[agent0].shape[0]
            )

            n_actions = env.action_space(agent0).n
            actor = load_actor(actor_path, obs_spec, n_actions, obs_mode, device)

            # ============================
            # 100-episode averaged stats
            # ============================
            stats = run_n_episodes(env, actor, n=100, device=device)
            results.append((grid, nagents, nitems, stats))

            writer.writerow([
                grid, nagents, nitems,
                stats["completion_mean"], stats["completion_std"],
                stats["timesteps_mean"], stats["timesteps_std"],
                stats["reward_ratio_mean"], stats["reward_ratio_std"],
                stats["mpd_mean"], stats["mpd_std"],
            ])

            # ============================
            # Record video for ONE episode
            # ============================

            video_path = f"videos/{obs_mode}_{grid}x{grid}_{nagents}A_{nitems}I.mp4"
            print(f"Recording video: {video_path}")

            with imageio.get_writer(video_path, fps=10, codec="libx264") as video_writer:
                run_episode(env, actor, device=device, video_writer=video_writer)

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_policy(
        obs_mode="knn",
        actor_path="mappo_knn_actor.pth",
        csv_path="replay_knn_results.csv",
        device=device
    )

    evaluate_policy(
        obs_mode="window",
        actor_path="mappo_window_actor.pth",
        csv_path="replay_window_results.csv",
        device=device
    )
