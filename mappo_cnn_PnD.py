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


# =========================
# Utils
# =========================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_global_state(env) -> np.ndarray:
    """
    Build global spatial state tensor for the convolutional critic.

    Shape: (grid_size, grid_size, C_global)

    Channels:
    0: all agents
    1: all delivery points
    2 + i: item i position (if not delivered)
    2 + num_items + i: delivery target for item i
    """
    G = env.grid_size
    N_items = len(env.items)

    C = 2 + 2 * N_items
    state = np.zeros((G, G, C), dtype=np.float32)

    # ---- Channel 0: Agents ----
    for agent in env.possible_agents:
        ax, ay = env.agent_location[agent]
        state[ax, ay, 0] = 1.0

    # ---- Channel 1: Delivery points ----
    for dp in env.delivery_points:
        dx, dy = env.delivery_locations[dp]
        state[dx, dy, 1] = 1.0

    # ---- Item channels (2 ... 2+N_items-1) ----
    for i, item in enumerate(env.items):
        if env.delivered[item]:
            continue
        ix, iy = env.item_locations[item]
        state[ix, iy, 2 + i] = 1.0

    # ---- Delivery target channels (2+N_items ... 2+2*N_items-1) ----
    for i, item in enumerate(env.items):
        dp = env.delivery_correspondence[item]  # e.g., "delivery_1"
        dx, dy = env.delivery_locations[dp]
        state[dx, dy, 2 + N_items + i] = 1.0

    return state


# =========================
# Networks
# =========================

class SharedActor(nn.Module):
    """
    CNN-based actor for egocentric grid observations.
    Input:
      - image: (B, H, W, C_obs)
      - carry: (B, 1) scalar flag
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, hidden: int = 128):
        super().__init__()

        H, W, C = obs_shape  # image shape from env["image"]

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_dim = self.cnn(dummy).view(1, -1).shape[1]

        # +1 for carry flag
        self.fc = nn.Sequential(
            nn.LayerNorm(conv_dim + 1),
            nn.Linear(conv_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, image: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
        """
        image: (B, H, W, C)
        carry: (B, 1)
        """
        x = image.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, carry], dim=1)
        logits = self.fc(x)
        return logits


class ConvCentralCritic(nn.Module):
    """
    Convolutional centralized critic V(s) with global spatial tensor.
    Input: state_img (B, H, W, C_global)
    """
    def __init__(self, grid_size: int, in_channels: int, hidden: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size, grid_size)
            conv_dim = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.LayerNorm(conv_dim),
            nn.Linear(conv_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_img: torch.Tensor) -> torch.Tensor:
        """
        state_img: (B, H, W, C_global)
        """
        x = state_img.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        v = self.fc(x).squeeze(-1)
        return v


# =========================
# Rollout Buffer
# =========================

@dataclass
class Transition:
    image: np.ndarray      # local obs image
    carry: np.ndarray      # local carry flag
    state_img: np.ndarray  # global state image (H,W,C_global)
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
        Compute GAE advantages and returns per agent (in-place).
        """
        self.advantages: Dict[str, np.ndarray] = {}
        self.returns: Dict[str, np.ndarray] = {}

        for a in self.agent_order:
            traj = self.storage[a]
            T = len(traj)
            adv = np.zeros(T, dtype=np.float32)

            next_adv = 0.0
            next_value = last_values[a]

            for t in reversed(range(T)):
                done = traj[t].done
                mask = 0.0 if done else 1.0
                delta = traj[t].reward + gamma * next_value * mask - traj[t].value
                next_adv = delta + gamma * lam * mask * next_adv
                adv[t] = next_adv
                next_value = traj[t].value

            values = np.array([tr.value for tr in traj], dtype=np.float32)
            ret = adv + values

            self.advantages[a] = adv
            self.returns[a] = ret

    def get_flat_batches(self):
        """
        Concatenate trajectories of all agents into flat arrays.
        """
        imgs, carries, states = [], [], []
        acts, logps, vals, advs, rets = [], [], [], [], []

        for a in self.agent_order:
            traj = self.storage[a]

            imgs.append(np.stack([tr.image for tr in traj]))
            carries.append(np.stack([tr.carry for tr in traj]))
            states.append(np.stack([tr.state_img for tr in traj]))

            acts.append(np.array([tr.action for tr in traj], dtype=np.int64))
            logps.append(np.array([tr.logp for tr in traj], dtype=np.float32))
            vals.append(np.array([tr.value for tr in traj], dtype=np.float32))
            advs.append(self.advantages[a])
            rets.append(self.returns[a])

        images = np.concatenate(imgs, axis=0)
        carries = np.concatenate(carries, axis=0)
        states = np.concatenate(states, axis=0)
        acts = np.concatenate(acts, axis=0)
        logps = np.concatenate(logps, axis=0)
        vals = np.concatenate(vals, axis=0)
        advs = np.concatenate(advs, axis=0)
        rets = np.concatenate(rets, axis=0)

        return images, carries, states, acts, logps, vals, advs, rets


# =========================
# MAPPO Trainer
# =========================

class MAPPO:
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        n_actions: int,
        grid_size: int,
        num_items: int,
        num_agents: int,
        device: str = "cpu",
        hidden_actor: int = 128,
        hidden_critic: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.device = device
        self.num_agents = num_agents

        # Actor
        self.actor = SharedActor(obs_shape, n_actions, hidden_actor).to(device)

        # Critic uses spatial global state with item-wise channels
        critic_in_channels = 2 + 2 * num_items  # as in stack_global_state
        self.critic = ConvCentralCritic(
            grid_size=grid_size,
            in_channels=critic_in_channels,
            hidden=hidden_critic
        ).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, obs_dict: Dict, state_img: np.ndarray):
        """
        obs_dict: {"image": (H,W,C_obs), "carry": (1,)}
        state_img: (G,G,C_global)
        """
        image = torch.tensor(
            obs_dict["image"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1,H,W,C_obs)
        carry = torch.tensor(
            obs_dict["carry"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1,1)

        s = torch.tensor(
            state_img, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1,G,G,C_global)

        logits = self.actor(image, carry)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)

        v = self.critic(s)

        return int(a.item()), float(logp.item()), float(v.item())

    def update(
        self,
        buffer: RolloutBuffer,
        epochs: int,
        minibatch_size: int,
        gamma: float,
        lam: float,
        writer: SummaryWriter,
        global_step: int,
    ):
        images, carries, states, acts, old_logps, old_vals, advs, rets = buffer.get_flat_batches()

        # normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        img_t = torch.tensor(images, dtype=torch.float32, device=self.device)
        carry_t = torch.tensor(carries, dtype=torch.float32, device=self.device)
        state_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(acts, dtype=torch.int64, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(rets, dtype=torch.float32, device=self.device)

        N = img_t.shape[0]
        idxs = np.arange(N)

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch_size):
                mb = idxs[start:start + minibatch_size]

                mb_img = img_t[mb]
                mb_carry = carry_t[mb]
                mb_state = state_t[mb]
                mb_acts = acts_t[mb]
                mb_old_logps = old_logps_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]

                # ---- Actor update ----
                logits = self.actor(mb_img, mb_carry)
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

                # ---- Critic update ----
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
    render_every: int = 0,
):
    agent_order = env.possible_agents[:]
    num_agents = len(agent_order)
    num_items = len(env.items)
    grid_size = env.grid_size

    # Infer obs shape from dict["image"]
    dummy_obs, _ = env.reset()
    obs_shape = dummy_obs[agent_order[0]]["image"].shape  # (H,W,C_obs)
    n_actions = env.action_space(agent_order[0]).n

    algo = MAPPO(
        obs_shape=obs_shape,
        n_actions=n_actions,
        grid_size=grid_size,
        num_items=num_items,
        num_agents=num_agents,
        device=device,
    )

    buffer = RolloutBuffer(agent_order)
    writer = SummaryWriter(log_dir="runs/mappo_area_PnD")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0
    step_count = 0
    episode = 0

    pbar = tqdm.tqdm(total=total_episodes)

    while episode < total_episodes:
        buffer.clear()

        for t in range(rollout_len):
            state_img = stack_global_state(env)  # (G,G,C_global)

            actions = {}
            logps = {}
            values = {}

            for a in agent_order:
                act, logp, val = algo.act(obs[a], state_img)
                actions[a] = act
                logps[a] = logp
                values[a] = val

            next_obs, rewards, dones, truncs, infos = env.step(actions)

            for a in agent_order:
                buffer.add(a, Transition(
                    image=obs[a]["image"],
                    carry=obs[a]["carry"],
                    state_img=state_img,
                    action=actions[a],
                    logp=logps[a],
                    value=values[a],
                    reward=rewards[a],
                    done=dones[a] or truncs[a],
                ))

                ep_return += rewards[a]
                ep_len += 1

            obs = next_obs
            step_count += num_agents

            if render_every > 0 and (episode % render_every == 0):
                env.render()

            if all(dones.values()) or all(truncs.values()):
                writer.add_scalar("episode/return", ep_return, global_step=episode)
                writer.add_scalar("episode/length", ep_len, global_step=episode)
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0
                episode += 1
                pbar.update(1)

        # Bootstrap last values for GAE
        last_state_img = stack_global_state(env)
        last_s_t = torch.tensor(last_state_img, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            v_last = algo.critic(last_s_t).item()
        last_values = {a: v_last for a in agent_order}

        buffer.compute_gae(gamma=gamma, lam=lam, last_values=last_values)

        algo.update(
            buffer=buffer,
            epochs=update_epochs,
            minibatch_size=minibatch_size,
            gamma=gamma,
            lam=lam,
            writer=writer,
            global_step=step_count,
        )

    pbar.close()
    env.close()
    writer.close()
    return algo


# =========================
# Main
# =========================

if __name__ == "__main__":
    from env_PnD_area import PickupDelivery

    set_seed(0)

    env = PickupDelivery(
        grid_size=7,
        num_agents=3,
        num_items=4,
        num_delivery_points=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    algo = train_mappo(
        env=env,
        total_episodes=5000,
        rollout_len=128,
        update_epochs=4,
        minibatch_size=256,
        gamma=0.99,
        lam=0.95,
        device=device,
        render_every=0,  # set >0 if you want to watch some episodes
    )

    torch.save(algo.actor.state_dict(), "mappo_cnn_PnD_actor.pth")
