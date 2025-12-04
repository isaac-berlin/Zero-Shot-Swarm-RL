import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from foraging import Foraging


# ============================================================
# Utils
# ============================================================

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_global_state(env) -> np.ndarray:
    """
    Global CTDE state: concat all agent and item coordinates.
    Matches logic from all 3 original scripts.
    """
    state = []
    for agent in env.possible_agents:
        state.extend(env.agent_location[agent])
    for item in env.items:
        state.extend(env.item_locations[item])
    return np.array(state, dtype=np.float32)


# ============================================================
# Networks: Unified Actor for 3 Observation Types
# ============================================================

class ActorMLP(nn.Module):
    """Used for obs_mode = vector or obs_mode = knn."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCNN(nn.Module):
    """Used for obs_mode = window (CNN)."""
    def __init__(self, obs_shape, n_actions, hidden=128):
        super().__init__()

        H, W, C = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )

        # compute conv output dimension
        with torch.no_grad():
            x = torch.zeros(1, C, H, W)
            conv_out = self.cnn(x).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.LayerNorm(conv_out),
            nn.Linear(conv_out, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs):
        # obs: (B, H, W, C) → (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
        x = self.cnn(obs)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class CentralCritic(nn.Module):
    """Shared critic across modes."""
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ============================================================
# Rollout Buffer
# ============================================================

@dataclass
class Transition:
    obs: np.ndarray
    state: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self, agent_order):
        self.agent_order = agent_order
        self.storage = {a: [] for a in agent_order}

    def add(self, agent, tr):
        self.storage[agent].append(tr)

    def clear(self):
        for a in self.agent_order:
            self.storage[a].clear()

    def compute_gae(self, gamma, lam, last_values):
        self.advantages, self.returns = {}, {}

        for a in self.agent_order:
            traj = self.storage[a]
            T = len(traj)

            adv = np.zeros(T, np.float32)
            ret = np.zeros(T, np.float32)

            next_adv = 0
            next_value = last_values[a]

            for t in reversed(range(T)):
                done = traj[t].done
                mask = 0 if done else 1
                delta = traj[t].reward + gamma * next_value * mask - traj[t].value
                next_adv = delta + gamma * lam * mask * next_adv
                adv[t] = next_adv
                next_value = traj[t].value

            ret = adv + np.array([tr.value for tr in traj], np.float32)

            self.advantages[a] = adv
            self.returns[a] = ret

    def get_flat_batches(self):
        obs_list, state_list, act_list = [], [], []
        logp_list, val_list, adv_list, ret_list = [], [], [], []

        for a in self.agent_order:
            traj = self.storage[a]
            obs_list.append(np.stack([tr.obs for tr in traj]))
            state_list.append(np.stack([tr.state for tr in traj]))
            act_list.append(np.array([tr.action for tr in traj]))
            logp_list.append(np.array([tr.logp for tr in traj], np.float32))
            val_list.append(np.array([tr.value for tr in traj], np.float32))
            adv_list.append(self.advantages[a])
            ret_list.append(self.returns[a])

        obs = np.concatenate(obs_list)
        state = np.concatenate(state_list)
        acts = np.concatenate(act_list)
        logps = np.concatenate(logp_list)
        vals = np.concatenate(val_list)
        advs = np.concatenate(adv_list)
        rets = np.concatenate(ret_list)

        return obs, state, acts, logps, vals, advs, rets


# ============================================================
# Unified MAPPO
# ============================================================

class MAPPO:
    def __init__(self, obs_spec, n_actions, state_dim, num_agents, obs_mode, device="cpu"):
        """
        obs_spec: obs_shape (H,W,C) or obs_dim for vector/knn
        """
        self.device = device
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.mode = obs_mode

        # --- Build Actor depending on mode ---
        if obs_mode in ("vector", "knn"):
            obs_dim = obs_spec
            self.actor = ActorMLP(obs_dim, n_actions).to(device)
        elif obs_mode == "window":
            obs_shape = obs_spec
            self.actor = ActorCNN(obs_shape, n_actions).to(device)
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")

        # Central critic is shared
        self.critic = CentralCritic(state_dim).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

    @torch.no_grad()
    def act(self, obs, state):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.critic(state_t)
        return int(action.item()), float(logp.item()), float(value.item())

    def update(self, buffer, epochs, minibatch, gamma, lam, writer, global_step):
        obs, state, acts, old_logps, old_vals, advs, rets = buffer.get_flat_batches()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(acts, dtype=torch.int64, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(rets, dtype=torch.float32, device=self.device)

        N = obs_t.shape[0]
        idxs = np.arange(N)

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch):
                mb = idxs[start:start + minibatch]

                # mini-batch slice
                mb_obs = obs_t[mb]
                mb_state = state_t[mb]
                mb_acts = acts_t[mb]
                mb_old_logps = old_logps_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]

                # Actor update
                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - mb_old_logps)
                unclipped = ratio * mb_advs
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advs

                policy_loss = -torch.min(unclipped, clipped).mean()
                actor_loss = policy_loss - self.ent_coef * entropy

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # Critic update
                values = self.critic(mb_state)
                vf_loss = (mb_rets - values).pow(2).mean()

                critic_loss = self.vf_coef * vf_loss
                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                # Logging
                if writer:
                    writer.add_scalar("loss/policy", policy_loss.item(), global_step)
                    writer.add_scalar("loss/value", vf_loss.item(), global_step)
                    writer.add_scalar("loss/entropy", entropy.item(), global_step)


# ============================================================
# Unified Training Loop
# ============================================================

def train_mappo(
    env,
    total_episodes=3000,
    rollout_len=128,
    update_epochs=4,
    minibatch_size=256,
    gamma=0.99,
    lam=0.95,
    device="cpu",
    render_every=0,
):

    agent_order = env.possible_agents[:]
    num_agents = len(agent_order)

    # infer observation structure
    dummy_obs, _ = env.reset()
    sample_obs = dummy_obs[agent_order[0]]

    if env.obs_mode == "window":
        obs_spec = sample_obs.shape     # (H, W, C)
    else:
        obs_spec = sample_obs.shape[0]  # dim

    state_dim = stack_global_state(env).shape[0]
    n_actions = env.action_space(agent_order[0]).n

    algo = MAPPO(
        obs_spec=obs_spec,
        n_actions=n_actions,
        state_dim=state_dim,
        num_agents=num_agents,
        obs_mode=env.obs_mode,
        device=device,
    )

    buffer = RolloutBuffer(agent_order)

    writer = SummaryWriter(log_dir=f"runs/foraging_{env.obs_mode}")
    obs, _ = env.reset()

    ep_return, ep_len, step_count, episode = 0, 0, 0, 0

    pbar = tqdm.tqdm(total=total_episodes)
    while episode < total_episodes:
        buffer.clear()

        # rollout
        for _ in range(rollout_len):
            state = stack_global_state(env)

            actions, logps, values = {}, {}, {}

            for a in agent_order:
                act, logp, val = algo.act(obs[a], state)
                actions[a] = act
                logps[a] = logp
                values[a] = val

            next_obs, rewards, dones, truncs, infos = env.step(actions)

            for a in agent_order:
                buffer.add(
                    a,
                    Transition(
                        obs=obs[a],
                        state=state,
                        action=actions[a],
                        logp=logps[a],
                        value=values[a],
                        reward=rewards[a],
                        done=dones[a] or truncs[a],
                    ),
                )
                ep_return += rewards[a]
                ep_len += 1

            obs = next_obs
            step_count += num_agents

            if render_every > 0 and episode % render_every == 0:
                env.render()

            if all(dones.values()) or all(truncs.values()):
                writer.add_scalar("episode/return", ep_return, episode)
                writer.add_scalar("episode/length", ep_len, episode)
                obs, _ = env.reset()
                ep_return, ep_len = 0, 0
                episode += 1
                pbar.update(1)

        # bootstrap value
        final_state = stack_global_state(env)
        with torch.no_grad():
            v_last = algo.critic(
                torch.tensor(final_state, dtype=torch.float32, device=device).unsqueeze(0)
            ).item()

        last_vals = {a: v_last for a in agent_order}
        buffer.compute_gae(gamma, lam, last_vals)

        algo.update(
            buffer=buffer,
            epochs=update_epochs,
            minibatch=minibatch_size,
            gamma=gamma,
            lam=lam,
            writer=writer,
            global_step=step_count,
        )

    env.close()
    writer.close()
    return algo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = Foraging(
        grid_size=10,
        num_agents=4,
        num_items=4,
        obs_mode="knn",  # "vector", "window", "knn"
    )

    algo = train_mappo(env, total_episodes=2000, device=device)

    torch.save(algo.actor.state_dict(), f"mappo_{env.obs_mode}_actor.pth")
