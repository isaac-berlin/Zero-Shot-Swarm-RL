import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ---- import your ParallelEnv ----
# from your_env_file import PickupDelivery
# Ensure your env class name matches.
from env import PickupDelivery


# =========================
# Utils
# =========================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_global_state(obs_dict: Dict[str, np.ndarray], agent_order: List[str]) -> np.ndarray:
    """
    Global state for centralized critic: concat observations in fixed agent order.
    Shape: (num_agents * obs_dim,)
    """
    return np.concatenate([obs_dict[a] for a in agent_order], axis=0)


# =========================
# Networks
# =========================

class SharedActor(nn.Module):
    """
    Parameter-shared policy π(a|o) for all agents.
    Discrete actions -> Categorical distribution.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns logits
        return self.net(obs)


class CentralCritic(nn.Module):
    """
    Centralized V(s) where s is global state (concat of all agent obs).
    """
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)  # (batch,)


# =========================
# Rollout Buffer
# =========================

@dataclass
class Transition:
    obs: np.ndarray            # (obs_dim,)
    state: np.ndarray          # (state_dim,)
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self, agent_order: List[str]):
        self.agent_order = agent_order
        self.storage: Dict[str, List[Transition]] = {a: [] for a in agent_order}

    def add(self, agent: str, trans: Transition):
        self.storage[agent].append(trans)

    def clear(self):
        for a in self.agent_order:
            self.storage[a].clear()

    def compute_gae(self, gamma: float, lam: float, last_values: Dict[str, float]):
        """
        Adds advantages and returns to each Transition (in-place).
        """
        self.advantages: Dict[str, np.ndarray] = {}
        self.returns: Dict[str, np.ndarray] = {}

        for a in self.agent_order:
            traj = self.storage[a]
            T = len(traj)
            adv = np.zeros(T, dtype=np.float32)
            ret = np.zeros(T, dtype=np.float32)

            next_adv = 0.0
            next_value = last_values[a]

            for t in reversed(range(T)):
                done = traj[t].done
                mask = 0.0 if done else 1.0
                delta = traj[t].reward + gamma * next_value * mask - traj[t].value
                next_adv = delta + gamma * lam * mask * next_adv
                adv[t] = next_adv

                next_value = traj[t].value

            ret = adv + np.array([tr.value for tr in traj], dtype=np.float32)

            self.advantages[a] = adv
            self.returns[a] = ret

    def get_flat_batches(self):
        """
        Flatten all agents' trajectories into arrays for PPO updates.
        """
        obs_list, state_list, act_list, logp_list, val_list, adv_list, ret_list = [], [], [], [], [], [], []

        for a in self.agent_order:
            traj = self.storage[a]
            obs_list.append(np.stack([tr.obs for tr in traj]))
            state_list.append(np.stack([tr.state for tr in traj]))
            act_list.append(np.array([tr.action for tr in traj], dtype=np.int64))
            logp_list.append(np.array([tr.logp for tr in traj], dtype=np.float32))
            val_list.append(np.array([tr.value for tr in traj], dtype=np.float32))
            adv_list.append(self.advantages[a])
            ret_list.append(self.returns[a])

        obs = np.concatenate(obs_list, axis=0)
        state = np.concatenate(state_list, axis=0)
        acts = np.concatenate(act_list, axis=0)
        logps = np.concatenate(logp_list, axis=0)
        vals = np.concatenate(val_list, axis=0)
        advs = np.concatenate(adv_list, axis=0)
        rets = np.concatenate(ret_list, axis=0)

        return obs, state, acts, logps, vals, advs, rets


# =========================
# MAPPO Trainer
# =========================

class MAPPO:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        num_agents: int,
        hidden_actor: int = 128,
        hidden_critic: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = obs_dim * num_agents
        self.n_actions = n_actions

        self.actor = SharedActor(obs_dim, n_actions, hidden_actor).to(device)
        self.critic = CentralCritic(self.state_dim, hidden_critic).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.critic(state_t)

        return int(action.item()), float(logp.item()), float(value.item())

    def update(
        self,
        buffer: RolloutBuffer,
        epochs: int = 4,
        minibatch_size: int = 256,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        obs, state, acts, old_logps, old_vals, advs, rets = buffer.get_flat_batches()

        # normalize advantages
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
            for start in range(0, N, minibatch_size):
                mb = idxs[start:start + minibatch_size]

                mb_obs = obs_t[mb]
                mb_state = state_t[mb]
                mb_acts = acts_t[mb]
                mb_old_logps = old_logps_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]

                # ---- Actor loss (PPO clipped) ----
                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - mb_old_logps)
                unclipped = ratio * mb_advs
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(unclipped, clipped).mean()

                actor_loss = policy_loss - self.ent_coef * entropy

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # ---- Critic loss ----
                values = self.critic(mb_state)
                vf_loss = (mb_rets - values).pow(2).mean()

                critic_loss = self.vf_coef * vf_loss

                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()


# =========================
# Training Loop
# =========================

def train_mappo(
    env,
    total_steps: int = 200_000,
    rollout_len: int = 128,
    update_epochs: int = 4,
    minibatch_size: int = 256,
    gamma: float = 0.99,
    lam: float = 0.95,
    device: str = "cpu",
    render_every: int = 0,  # set >0 to render occasionally
):
    agent_order = env.possible_agents[:]
    num_agents = len(agent_order)

    # infer dims
    dummy_obs, _ = env.reset()
    obs_dim = dummy_obs[agent_order[0]].shape[0]
    n_actions = env.action_space(agent_order[0]).n

    algo = MAPPO(
        obs_dim=obs_dim,
        n_actions=n_actions,
        num_agents=num_agents,
        device=device
    )

    buffer = RolloutBuffer(agent_order)

    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0
    episode = 0
    step_count = 0

    while step_count < total_steps:
        buffer.clear()

        # ===== rollout =====
        for t in range(rollout_len):
            state = stack_global_state(obs, agent_order)

            actions = {}
            logps = {}
            values = {}

            for a in agent_order:
                act, logp, val = algo.act(obs[a], state)
                actions[a] = act
                logps[a] = logp
                values[a] = val

            next_obs, rewards, dones, truncs, infos = env.step(actions)

            # store per-agent transition
            for a in agent_order:
                buffer.add(a, Transition(
                    obs=obs[a],
                    state=state,
                    action=actions[a],
                    logp=logps[a],
                    value=values[a],
                    reward=rewards[a],
                    done=dones[a] or truncs[a]
                ))

                ep_return += rewards[a]
                ep_len += 1

            obs = next_obs
            step_count += num_agents

            # optional render
            if render_every > 0 and (episode % render_every == 0):
                env.render()

            # if episode ended, reset
            if all(dones.values()) or all(truncs.values()):
                print(f"[episode {episode}] return={ep_return:.2f} len={ep_len}")
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0
                episode += 1

        # bootstrap last values for GAE
        last_values = {}
        last_state = stack_global_state(obs, agent_order)
        last_state_t = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            v_last = algo.critic(last_state_t).item()
        for a in agent_order:
            last_values[a] = v_last  # shared centralized value

        buffer.compute_gae(gamma=gamma, lam=lam, last_values=last_values)

        # ===== PPO update =====
        algo.update(
            buffer,
            epochs=update_epochs,
            minibatch_size=minibatch_size,
            gamma=gamma,
            lam=lam,
        )

    env.close()
    return algo


# =========================
# Main
# =========================

if __name__ == "__main__":
    set_seed(0)

    env = PickupDelivery(
        grid_size=7,
        num_agents=3,
        num_items=4,
        num_delivery_points=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    algo = train_mappo(
        env,
        total_steps=150_000_000_000,
        rollout_len=128,
        update_epochs=4,
        minibatch_size=256,
        gamma=0.99,
        lam=0.95,
        device=device,
        render_every=1000  # set to e.g. 20 to watch training
    )
