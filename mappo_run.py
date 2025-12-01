import torch
import numpy as np
from torch.distributions import Categorical

from env import PickupDelivery   # your parallel env
from mappo import SharedActor  # import the same class definition used in training


# -----------------------------
# Helper
# -----------------------------
def stack_global_state(obs_dict, agent_order):
    return np.concatenate([obs_dict[a] for a in agent_order], axis=0)


# -----------------------------
# Load Policy and Run Episode
# -----------------------------
def run_policy(
    actor_path: str = "mappo_actor.pth",
    grid_size=7,
    num_agents=3,
    num_items=4,
    num_delivery_points=2,
    render=True,
    stochastic=False,  # If True → sample; If False → argmax
    device="cpu"
):

    # ---- Create env ----
    env = PickupDelivery(
        grid_size=grid_size,
        num_agents=num_agents,
        num_items=num_items,
        num_delivery_points=num_delivery_points,
    )

    agent_order = env.possible_agents[:]

    # ---- Infer dims ----
    obs, _ = env.reset()
    obs_dim = obs[agent_order[0]].shape[0]
    n_actions = env.action_space(agent_order[0]).n

    # ---- Load Actor ----
    actor = SharedActor(obs_dim, n_actions)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.to(device)
    actor.eval()

    print(f"Loaded policy from: {actor_path}")

    # ---- Run ----
    episode = 0
    while True:
        obs, _ = env.reset()
        done_flags = {a: False for a in agent_order}
        total_reward = 0.0
        step = 0

        while not all(done_flags.values()):
            if render:
                env.render()

            # Build global state
            state = stack_global_state(obs, agent_order)
            state_t = torch.tensor(state, dtype=torch.float32, device=device)

            actions = {}

            for a in agent_order:
                o = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor(o)
                dist = Categorical(logits=logits)

                if stochastic:
                    act = dist.sample()
                else:
                    act = torch.argmax(logits, dim=-1)

                actions[a] = int(act.item())

            # Step env
            obs, rewards, dones, truncs, infos = env.step(actions)

            total_reward += sum(rewards.values())
            step += 1
            done_flags = {a: dones[a] or truncs[a] for a in agent_order}

        print(f"[Episode {episode}] Return: {total_reward:.2f}, Steps: {step}")
        episode += 1


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_policy(
        actor_path="mappo_actor.pth",
        render=True,
        stochastic=False,  # change to True for sampling
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
