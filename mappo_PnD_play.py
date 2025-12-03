import torch
from torch.distributions import Categorical

from env_PnD_area import PickupDelivery
from mappo_cnn_PnD import SharedActor   # <-- IMPORT the actor directly


@torch.no_grad()
def select_action(actor, obs_dict, device):
    """
    obs_dict = { "image": (H,W,C), "carry": (1,) }
    """
    image = torch.tensor(obs_dict["image"], dtype=torch.float32, device=device).unsqueeze(0)
    carry = torch.tensor(obs_dict["carry"], dtype=torch.float32, device=device).unsqueeze(0)

    logits = actor(image, carry)
    dist = Categorical(logits=logits)
    a = dist.sample()
    return int(a.item())


def run_policy(
    actor_path="mappo_cnn_PnD_actor.pth",
    episodes=5,
    grid_size=7,
    num_agents=3,
    num_items=4,
    num_delivery_points=2,
    device="cpu",
    render=True
):

    env = PickupDelivery(
        grid_size=grid_size,
        num_agents=num_agents,
        num_items=num_items,
        num_delivery_points=num_delivery_points
    )

    # Infer obs shape
    obs, _ = env.reset()
    example_agent = env.possible_agents[0]
    obs_shape = obs[example_agent]["image"].shape
    n_actions = env.action_space(example_agent).n

    # ---- Load Actor ----
    actor = SharedActor(obs_shape, n_actions).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print(f"Loaded actor weights from: {actor_path}")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = {a: False for a in env.possible_agents}
        ep_return = {a: 0.0 for a in env.possible_agents}

        while not all(done.values()):
            actions = {}
            for agent in env.possible_agents:
                actions[agent] = select_action(actor, obs[agent], device)

            obs_next, rewards, dones, truncs, infos = env.step(actions)

            for agent in env.possible_agents:
                ep_return[agent] += rewards[agent]
                done[agent] = dones[agent] or truncs[agent]

            obs = obs_next

            if render:
                env.render()

        print(f"Episode {ep+1} finished. Returns:", ep_return)

    env.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_policy(
        actor_path="mappo_cnn_PnD_actor.pth",
        episodes=10,
        device=device
    )
