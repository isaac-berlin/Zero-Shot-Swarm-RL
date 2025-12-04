import torch
import numpy as np
from torch.distributions import Categorical

from env_knn import PickupOnly
from mappo_knn import SharedActor
from mappo_knn import stack_global_state


# -----------------------------
# Load & Run Trained Policy
# -----------------------------
def run_policy(
    actor_path: str = "mappo_knn_actor.pth",
    grid_size=14,
    num_agents=5,
    num_items=7,
    render=True,  # If True, sample stochastically; If False, use argmax
    device="cpu"
):
    # ---- Create Environment ----
    env = PickupOnly(
        grid_size=grid_size,
        num_agents=num_agents,
        num_items=num_items,
    )

    agent_order = env.possible_agents[:]

    # ---- Infer observation dims ----
    obs, _ = env.reset()
    obs_vec = obs[agent_order[0]]        # shape = (obs_dim,)
    obs_dim = obs_vec.shape[0]

    # ---- Action space ----
    n_actions = env.action_space(agent_order[0]).n

    # ---- Load Actor ----
    actor = SharedActor(obs_dim=obs_dim, n_actions=n_actions)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.to(device)
    actor.eval()

    print(f"Loaded policy from: {actor_path}")

    # ---- Run Episodes Forever ----
    episode = 0
    while True:
        obs, _ = env.reset()
        done_flags = {a: False for a in agent_order}
        total_reward = 0.0
        steps = 0

        while not all(done_flags.values()):
            if render:
                env.render()

            # Build global state vector
            state = stack_global_state(env)
            state_t = torch.tensor(state, dtype=torch.float32, device=device)

            actions = {}

            for a in agent_order:
                obs_t = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)

                logits = actor(obs_t)
                dist = Categorical(logits=logits)

                if render:
                    # stochastic sample
                    act = dist.sample()
                else:
                    # greedy mode
                    act = torch.argmax(logits, dim=-1)

                actions[a] = int(act.item())

            # Environment step
            obs, rewards, dones, truncs, infos = env.step(actions)

            total_reward += sum(rewards.values())
            steps += 1

            # Update termination flags
            done_flags = {a: dones[a] or truncs[a] for a in agent_order}

        print(f"[Episode {episode}] Return: {total_reward:.2f}, Steps: {steps}")
        episode += 1


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_policy(
        actor_path="mappo_knn_actor.pth",
        render=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
