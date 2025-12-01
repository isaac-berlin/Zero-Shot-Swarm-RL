from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from pettingzoo.utils import ParallelEnv
from pettingzoo.test import parallel_api_test
from gymnasium import spaces
import pygame
import math


class PickupDelivery(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "PickupDeliveryEnv-v0",
    }
    
    def __init__(self, grid_size: int = 5, num_agents: int = 2, num_items: int = 3, num_delivery_points: int = 1):
        self.grid_size = grid_size
        self.n_agents = num_agents
        self.n_items = num_items
        self.n_delivery_points = num_delivery_points

        self.max_steps = 200
        self.timestep = 0
        
        # Items & deliveries
        self.items = [f"item_{i}" for i in range(num_items)]
        self.item_locations = {item: (0, 0) for item in self.items}
        self.delivered = {item: False for item in self.items}

        self.delivery_points = [f"delivery_{i}" for i in range(num_delivery_points)]
        self.delivery_locations = {p: (0, 0) for p in self.delivery_points}

        # Each item assigned to one delivery point
        self.delivery_correspondence = {item: None for item in self.items}

        # Agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_location = {agent: (0, 0) for agent in self.agents}
        self.agent_carrying = {agent: None for agent in self.agents}
        
        # Action space
        self.action_spaces = {
            agent: spaces.Discrete(7)  # 0 stay, 1 up, 2 down, 3 left, 4 right, 5 grab, 6 drop
            for agent in self.possible_agents
        }

        # Observation size
        obs_size = (
            2 +  # self loc
            1 +  # self carrying
            (num_agents - 1) * 3 +  # other agent loc + carry
            num_items * 2 +         # item locations
            num_delivery_points * 2 # delivery locations
        )

        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=grid_size, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Rendering (pygame)
        self.render_mode = "human"
        self._pygame_initialized = False
        self._cell_size = 64
        self._margin = 40
        self.delivery_colors = {}
        self.item_colors = {}
        self._screen = None
        self._clock = None
        self._font = None

        # Reward shaping trackers
        self.prev_item_dist = {}
        self.prev_delivery_dist = {}


    # ============================================================
    # Reset
    # ============================================================
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.delivered = {item: False for item in self.items}
        self.agents = self.possible_agents[:]

        # Assign each item to random delivery point
        for item in self.items:
            self.delivery_correspondence[item] = random.choice(self.delivery_points)
            
        # Colors
        base_colors = [
            (255, 80, 80), (255, 200, 0), (80, 255, 120),
            (200, 80, 255), (255, 150, 80)
        ]
        for i, p in enumerate(self.delivery_points):
            self.delivery_colors[p] = base_colors[i % len(base_colors)]

        for item in self.items:
            dp = self.delivery_correspondence[item]
            self.item_colors[item] = self.delivery_colors[dp]

        # Sample positions
        needed = self.n_items + self.n_delivery_points + self.n_agents
        positions = set()
        while len(positions) < needed:
            positions.add((
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ))
        positions = list(positions)
        idx = 0

        # Items
        for item in self.items:
            self.item_locations[item] = positions[idx]
            idx += 1

        # Delivery points
        for p in self.delivery_points:
            self.delivery_locations[p] = positions[idx]
            idx += 1

        # Agents
        for a in self.agents:
            self.agent_location[a] = positions[idx]
            self.agent_carrying[a] = None
            idx += 1

        # Reward shaping initialization
        self.prev_item_dist = {}
        self.prev_delivery_dist = {}

        for agent in self.agents:
            loc = self.agent_location[agent]
            self.prev_item_dist[agent] = self._dist_to_nearest_item(loc)
            self.prev_delivery_dist[agent] = None

        return self._get_observations(), {agent: {} for agent in self.agents}


    # ============================================================
    # Step
    # ============================================================
    def step(self, actions):
        self.timestep += 1
        rewards = {agent: -0.01 for agent in self.agents}  # base step penalty

        # Apply actions
        for agent, action in actions.items():

            # Movement
            if action in [0, 1, 2, 3, 4]:
                self.agent_location[agent] = self._move(self.agent_location[agent], action)

            # Grab
            elif action == 5:
                if self._attempt_grab(agent):
                    rewards[agent] += 0.01

            # Drop / delivery
            elif action == 6:
                delivered = self._attempt_delivery(agent)
                if delivered:
                    rewards[agent] += 1.0
                else:
                    self._attempt_drop(agent)

        # ================================================================
        # Reward Shaping
        # ================================================================
        for agent in self.agents:
            loc = self.agent_location[agent]
            carried = self.agent_carrying[agent]

            # ---------------------------
            # Case 1: Not carrying → move toward nearest item
            # ---------------------------
            if carried is None:
                curr_dist = self._dist_to_nearest_item(loc)
                prev_dist = self.prev_item_dist.get(agent, None)

                # SAFE INIT
                if prev_dist is None:
                    prev_dist = curr_dist

                delta = prev_dist - curr_dist
                rewards[agent] += 0.05 * delta

                # update trackers
                self.prev_item_dist[agent] = curr_dist
                self.prev_delivery_dist[agent] = None

            # ---------------------------
            # Case 2: Carrying → move toward correct delivery point
            # ---------------------------
            else:
                curr_dist = self._dist_to_delivery(loc, carried)
                prev_dist = self.prev_delivery_dist.get(agent, None)

                # SAFE INIT
                if prev_dist is None:
                    prev_dist = curr_dist

                delta = prev_dist - curr_dist
                rewards[agent] += 0.05 * delta

                self.prev_delivery_dist[agent] = curr_dist
                self.prev_item_dist[agent] = None

        # ================================================================
        # Termination
        # ================================================================
        done = all(self.delivered.values())
        truncated = self.timestep >= self.max_steps
        dones = {agent: done or truncated for agent in self.agents}
        trunc = {agent: truncated for agent in self.agents}

        observations = self._get_observations()

        if done or truncated:
            self.agents = []

        info = {agent: {} for agent in observations}
        return observations, rewards, dones, trunc, info


    # ============================================================
    # Utility functions
    # ============================================================
    def _move(self, loc, action):
        x, y = loc
        if action == 1: y = min(self.grid_size - 1, y + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: x = max(0, x - 1)
        elif action == 4: x = min(self.grid_size - 1, x + 1)
        return (x, y)

    def _attempt_grab(self, agent):
        if self.agent_carrying[agent] is not None:
            return 0
        loc = self.agent_location[agent]
        for item, pos in self.item_locations.items():
            if pos == loc and not self.delivered[item]:
                self.agent_carrying[agent] = item
                return 1
        return 0

    def _attempt_drop(self, agent):
        item = self.agent_carrying[agent]
        if item is None:
            return 0
        self.item_locations[item] = self.agent_location[agent]
        self.agent_carrying[agent] = None
        return 1

    def _attempt_delivery(self, agent):
        item = self.agent_carrying[agent]
        if item is None:
            return 0

        if self.delivered[item]:
            return 0

        loc = self.agent_location[agent]
        target_point = self.delivery_correspondence[item]
        target_loc = self.delivery_locations[target_point]

        if loc == target_loc:
            self.delivered[item] = True
            self.agent_carrying[agent] = None
            return 1
        return 0

    # ============================
    # Distance helpers
    # ============================
    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _dist_to_nearest_item(self, loc):
        dists = [self._manhattan(loc, self.item_locations[item])
                 for item in self.items if not self.delivered[item]]
        return min(dists) if dists else 0

    def _dist_to_delivery(self, loc, item):
        point = self.delivery_correspondence[item]
        target = self.delivery_locations[point]
        return self._manhattan(loc, target)

    # ============================
    # Observations
    # ============================
    def _encode_carry(self, item):
        return -1 if item is None else int(item.split("_")[1])

    def _get_observations(self):
        return {agent: self._single_observation(agent) for agent in self.agents}

    def _single_observation(self, agent):
        self_loc = np.array(self.agent_location[agent], dtype=np.float32)
        self_car = np.array([self._encode_carry(self.agent_carrying[agent])], dtype=np.float32)

        other_info = []
        for other in self.agents:
            if other == agent:
                continue
            loc = self.agent_location[other]
            car = self._encode_carry(self.agent_carrying[other])
            other_info.extend([loc[0], loc[1], car])
        other_info = np.array(other_info, dtype=np.float32)

        item_info = []
        for item in self.items:
            ix, iy = self.item_locations[item]
            item_info.extend([ix, iy])
        item_info = np.array(item_info, dtype=np.float32)

        deliv_info = []
        for p in self.delivery_points:
            dx, dy = self.delivery_locations[p]
            deliv_info.extend([dx, dy])
        deliv_info = np.array(deliv_info, dtype=np.float32)

        return np.concatenate([self_loc, self_car, other_info, item_info, deliv_info]).astype(np.float32)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ============================================================
    # Rendering
    # ============================================================
    def render(self):
        if self.render_mode != "human":
            return

        self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return

        self._screen.fill((30, 30, 30))

        # Grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                sx, sy = self._grid_to_screen(x, y)
                rect = pygame.Rect(sx, sy, self._cell_size, self._cell_size)
                pygame.draw.rect(self._screen, (60, 60, 60), rect, width=1)

        # Delivery points
        for p in self.delivery_points:
            dx, dy = self.delivery_locations[p]
            sx, sy = self._grid_to_screen(dx, dy)
            rect = pygame.Rect(sx+4, sy+4, self._cell_size-8, self._cell_size-8)
            pygame.draw.rect(self._screen, self.delivery_colors[p], rect)
            pygame.draw.rect(self._screen, (255,255,255), rect, width=2)

        # Items
        for item in self.items:
            if self.delivered[item]:
                continue
            ix, iy = self.item_locations[item]
            sx, sy = self._grid_to_screen(ix, iy)
            center = (sx + self._cell_size//2, sy + self._cell_size//2)
            pygame.draw.circle(self._screen, self.item_colors[item], center, self._cell_size//4)

        # Agents
        agent_color = (0, 200, 255)
        for agent in self.agents:
            ax, ay = self.agent_location[agent]
            sx, sy = self._grid_to_screen(ax, ay)
            rect = pygame.Rect(sx+8, sy+8, self._cell_size-16, self._cell_size-16)
            pygame.draw.rect(self._screen, agent_color, rect)

            if self.agent_carrying[agent] is not None:
                pygame.draw.rect(self._screen, (255,255,255), rect, width=2)

        # timestep text
        text = self._font.render(f"t = {self.timestep}", True, (220,220,220))
        self._screen.blit(text, (10, 10))

        pygame.display.flip()
        self._clock.tick(10)

    def _grid_to_screen(self, x, y):
        sx = self._margin + x * self._cell_size
        sy = self._margin + (self.grid_size - 1 - y) * self._cell_size
        return sx, sy

    def _init_pygame(self):
        if self._pygame_initialized:
            return
        pygame.init()
        pygame.display.set_caption("PickupDeliveryEnv")

        width = self.grid_size * self._cell_size + 2 * self._margin
        height = self.grid_size * self._cell_size + 2 * self._margin

        self._screen = pygame.display.set_mode((width, height))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 18)

        self._pygame_initialized = True

    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False



# Test when running directly
if __name__ == "__main__":
    env = PickupDelivery(grid_size=7, num_agents=3, num_items=4, num_delivery_points=2)

    parallel_api_test(env, num_cycles=500)

    obs, _ = env.reset()

    for _ in range(200):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rew, done, trunc, info = env.step(actions)
        env.render()
        if all(done.values()) or all(trunc.values()):
            break

    env.close()
