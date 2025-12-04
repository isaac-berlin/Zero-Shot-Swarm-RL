import math
import os
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


from env_area import PickupOnly


# =========================
# Utils
# =========================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_global_state(env) -> np.ndarray:
    """
    location of all agents and items concatenated.
    """
    state = []
    for agent in env.possible_agents:
        state.append(env.agent_location[agent][0])  # x
        state.append(env.agent_location[agent][1])  # y
        
    for item in env.items:
        state.append(env.item_locations[item][0])  # x
        state.append(env.item_locations[item][1])  # y

    return np.array(state, dtype=np.float32)


# =========================
# Networks
# =========================

class SharedActor(nn.Module):
    """
    CNN-based actor for egocentric grid observations.
    Input: (H, W, C) but PyTorch wants (C, H, W).
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, hidden: int = 128):
        super().__init__()

        C = obs_shape[2]   # number of channels (e.g., 3)
        H = obs_shape[0]
        W = obs_shape[1]

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # compute CNN output size
        with torch.no_grad():
            x = torch.zeros(1, C, H, W)
            conv_out_dim = self.cnn(x).view(1, -1).shape[1]

        # MLP head
        self.fc = nn.Sequential(
            nn.LayerNorm(conv_out_dim),
            nn.Linear(conv_out_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, H, W, C)
        """
        # convert to (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)

        x = self.cnn(obs)
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x)
        return logits

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
        return self.net(state).squeeze(-1)


# =========================
# Rollout Buffer
# =========================

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
        obs_shape: Tuple[int, int, int],
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
        self.obs_shape = obs_shape
        self.state_dim = 14
        self.n_actions = n_actions

        self.actor = SharedActor(obs_shape, n_actions, hidden_actor).to(device)
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
        writer=None,
        global_step=None,
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
                
                if writer is not None:
                    writer.add_scalar("loss/policy_loss", policy_loss.item(), global_step)
                    writer.add_scalar("loss/value_loss", vf_loss.item(), global_step)
                    writer.add_scalar("loss/entropy", entropy.item(), global_step)
                    writer.add_scalar("loss/actor_lr", self.opt_actor.param_groups[0]['lr'], global_step)
                    writer.add_scalar("loss/critic_lr", self.opt_critic.param_groups[0]['lr'], global_step)


# =========================
# Training Loop
# =========================

def train_mappo(
    env,
    total_episodes: int = 5000,
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
    obs_shape = dummy_obs[agent_order[0]].shape
    n_actions = env.action_space(agent_order[0]).n

    algo = MAPPO(
        obs_shape=obs_shape,
        n_actions=n_actions,
        num_agents=num_agents,
        device=device
    )

    buffer = RolloutBuffer(agent_order)
    
    writer = SummaryWriter(log_dir="runs/mappo_area")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0
    step_count = 0
    episode = 0

    pbar = tqdm.tqdm(total=total_episodes)
    while episode < total_episodes:
        buffer.clear()

        # ===== rollout =====
        for t in range(rollout_len):
            state = stack_global_state(env)

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
                writer.add_scalar("episode/return", ep_return, global_step=episode)
                writer.add_scalar("episode/length", ep_len, global_step=episode)
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0
                episode += 1
                pbar.update(1)

        # bootstrap last values for GAE
        last_values = {}
        last_state = stack_global_state(env)
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
            writer=writer,
            global_step=step_count
        )

    pbar.close()
    env.close()
    writer.close() 
    return algo


# =========================
# Main
# =========================

if __name__ == "__main__":
    set_seed(0)

    env = PickupOnly(
        grid_size=7,
        num_agents=3,
        num_items=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    algo = train_mappo(
        env,
        total_episodes=10000,
        rollout_len=128,
        update_epochs=4,
        minibatch_size=256,
        gamma=0.99,
        lam=0.95,
        device=device,
        render_every=100  # set to e.g. 20 to watch training
    )
    
    torch.save(algo.actor.state_dict(), "mappo_area_actor.pth")
