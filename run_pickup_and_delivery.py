import torch
import numpy as np
from torch.distributions import Categorical

from pickup_and_delivery import PickupDelivery
from train_pickup_and_delivery import (
    ActorMLP,
    SharedActorCNN,
    stack_global_state,
)


# ======================================================================
# Helper: load correct actor type depending on obs_mode
# ======================================================================

def load_actor(actor_path, obs_mode, obs_sample, n_actions, device):
    """
    Automatically loads the correct actor architecture:

    - obs_mode = "vector" → ActorMLP
    - obs_mode = "knn"    → ActorMLP
    - obs_mode = "window" → SharedActorCNN
    """
    if obs_mode == "window":
        obs_shape = obs_sample["image"].shape  # (H,W,C)
        actor = SharedActorCNN(obs_shape, n_actions)
    else:
        obs_dim = obs_sample.shape[0]
        actor = ActorMLP(obs_dim, n_actions)

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.to(device)
    actor.eval()

    print(f"Loaded {obs_mode} actor from: {actor_path}")
    return actor


# ======================================================================
# Unified PnD Run Function
# ======================================================================

def run_policy(
    actor_path: str,
    grid_size=7,
    num_agents=3,
    num_items=4,
    num_delivery_points=2,
    obs_mode: str = "window",        # "vector", "window", or "knn"
    stochastic: bool = True,
    device: str = "cpu",
):
    """
    Unified policy execution loop for all Pickup-Delivery observation modes.
    """

    # -----------------------------------------------------
    # Create env
    # -----------------------------------------------------
    env = PickupDelivery(
        grid_size=grid_size,
        num_agents=num_agents,
        num_items=num_items,
        num_delivery_points=num_delivery_points,
        obs_mode=obs_mode,
    )

    agent_order = env.possible_agents[:]

    # -----------------------------------------------------
    # Infer dimensions
    # -----------------------------------------------------
    obs, _ = env.reset()
    sample_obs = obs[agent_order[0]]
    n_actions = env.action_space(agent_order[0]).n

    # -----------------------------------------------------
    # Load correct actor
    # -----------------------------------------------------
    actor = load_actor(actor_path, obs_mode, sample_obs, n_actions, device)

    # -----------------------------------------------------
    # Run forever
    # -----------------------------------------------------
    episode = 0
    while True:
        obs, _ = env.reset()
        done_flags = {a: False for a in agent_order}

        total_reward = 0.0
        steps = 0

        print(f"\n=== Episode {episode} ===")

        while not all(done_flags.values()):
            env.render()

            # Build global state tensor for critic
            state_img = stack_global_state(env)
            s_t = torch.tensor(state_img, dtype=torch.float32, device=device).unsqueeze(0)

            actions = {}

            for a in agent_order:
                if obs_mode == "window":
                    # Window input → separate image + carry
                    image = torch.tensor(
                        obs[a]["image"], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    carry = torch.tensor(
                        obs[a]["carry"], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    logits = actor(image, carry)
                else:
                    o_t = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
                    logits = actor(o_t)

                dist = Categorical(logits=logits)

                if stochastic:
                    act = dist.sample()
                else:
                    act = torch.argmax(logits, dim=-1)

                actions[a] = int(act.item())

            # Step
            obs, rewards, dones, truncs, infos = env.step(actions)

            total_reward += sum(rewards.values())
            steps += 1
            done_flags = {a: dones[a] or truncs[a] for a in agent_order}

        print(f"Episode {episode} finished | Return: {total_reward:.2f} | Steps: {steps}")
        episode += 1


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_policy(
        actor_path="mappo_pnd_knn_actor.pth",  # change to your saved model
        grid_size=7,
        num_agents=3,
        num_items=4,
        num_delivery_points=2,
        obs_mode="knn",                        # "vector", "window", "knn"
        stochastic=True,
        device=device,
    )
