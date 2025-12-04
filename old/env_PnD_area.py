from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from pettingzoo.utils import ParallelEnv, AECEnv
from pettingzoo.test import api_test, parallel_api_test
from gymnasium import spaces
import pygame



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

        # Each item is assigned to exactly one delivery point
        self.delivery_correspondence = {item: None for item in self.items}

        # Agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_location = {agent: (0, 0) for agent in self.agents}
        self.agent_carrying = {agent: None for agent in self.agents}
        
        # Action space: 0=up,1=down,2=left,3=right,4=interact
        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # === PickupOnly-style egocentric observation (plus carry flag) ===
        self.obs_radius = 3
        window = self.obs_radius * 2 + 1

        # Image → (window, window, 4)
        # 4 channels:
        # 0 = ego agent
        # 1 = other agents
        # 2 = items (undelivered)
        # 3 = delivery point of the item being carried
        
        # carry  → scalar 0 or 1
        self.observation_spaces = {
            agent: spaces.Dict({
                "image": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(window, window, 4),
                    dtype=np.float32
                ),
                "carry": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32
                )
            })
            for agent in self.possible_agents
        }
        
        # ----- Rendering (pygame) -----
        self.render_mode = "human"
        self._pygame_initialized = False
        self._cell_size = 64  # pixels per grid cell
        self._margin = 40     # border around grid
        self.delivery_colors = {}  # delivery point -> (r,g,b)
        self.item_colors = {}      # item -> (r,g,b)

        # Will be created lazily
        self._screen = None
        self._clock = None
        self._font = None

    # -------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.delivered = {item: False for item in self.items}
        self.agents = self.possible_agents[:]

        # Assign each item to a random delivery point
        for item in self.items:
            self.delivery_correspondence[item] = random.choice(self.delivery_points)
            
        base_colors = [
            (255, 80, 80),    # red
            (255, 200, 0),    # yellow
            (80, 255, 120),   # green
            (200, 80, 255),   # purple
            (255, 150, 80),   # orange
        ]

        for i, p in enumerate(self.delivery_points):
            self.delivery_colors[p] = base_colors[i % len(base_colors)]

        # ---- Assign item color = its delivery point color ----
        for item in self.items:
            dp = self.delivery_correspondence[item]
            self.item_colors[item] = self.delivery_colors[dp]

        # Sample unique locations
        needed = self.n_items + self.n_delivery_points + self.n_agents
        positions = set()

        while len(positions) < needed:
            positions.add((
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ))

        positions = list(positions)
        idx = 0

        # Place items
        for item in self.items:
            self.item_locations[item] = positions[idx]
            idx += 1

        # Place delivery points
        for p in self.delivery_points:
            self.delivery_locations[p] = positions[idx]
            idx += 1

        # Place agents
        for a in self.agents:
            self.agent_location[a] = positions[idx]
            self.agent_carrying[a] = None
            idx += 1

        return self._get_observations(), {agent: {} for agent in self.agents}

    # -------------------------------------------------------
    def step(self, actions):
        self.timestep += 1

        rewards = {agent: -0.01 for agent in self.agents}  # step penalty

        for agent, action in actions.items():

            # ------------------------
            # Movement 0–3
            # ------------------------
            if action in [0, 1, 2, 3]:
                self.agent_location[agent] = self._move(self.agent_location[agent], action)

            # ------------------------
            # Interact = action 4
            # ------------------------
            elif action == 4:
                carried = self.agent_carrying[agent]
                agent_loc = self.agent_location[agent]

                # If hands empty → try pickup
                if carried is None:
                    if self._attempt_pickup(agent):
                        rewards[agent] += 0.5
                        continue

                # If carrying → first try delivery, then fallback to drop
                else:
                    if self._attempt_delivery(agent):
                        rewards[agent] += 5.0
                        continue
                    else:
                        # Drop the item on the ground
                        if self._attempt_drop(agent):
                            rewards[agent] -= 0.1
                            continue

        # --- Termination ---
        done = all(self.delivered.values())
        truncated = self.timestep >= self.max_steps
        dones = {agent: done or truncated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        observations = self._get_observations()
        if done or truncated:
            self.agents = []
        
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, truncations, infos
    # -------------------------------------------------------
    def _move(self, loc, action):
        x, y = loc
        if action == 0:      # up
            y = min(self.grid_size - 1, y + 1)
        elif action == 1:    # down
            y = max(0, y - 1)
        elif action == 2:    # left
            x = max(0, x - 1)
        elif action == 3:    # right
            x = min(self.grid_size - 1, x + 1)
        return (x, y)

    # -------------------------------------------------------
    def _attempt_pickup(self, agent):
        """Pick up item at agent location if hands empty and item not delivered."""
        if self.agent_carrying[agent] is not None:
            return 0

        ax, ay = self.agent_location[agent]

        for item, (ix, iy) in self.item_locations.items():
            if not self.delivered[item] and (ix == ax and iy == ay):
                self.agent_carrying[agent] = item
                return 1

        return 0

    # -------------------------------------------------------
    def _attempt_drop(self, agent):
        carried = self.agent_carrying[agent]
        if carried is None:
            return 0

        self.item_locations[carried] = self.agent_location[agent]
        self.agent_carrying[agent] = None
        return 1
    
    # -------------------------------------------------------
    def _attempt_delivery(self, agent):
        """Deliver only correct item → correct delivery point."""
        carried = self.agent_carrying[agent]
        if carried is None:
            return 0

        if self.delivered[carried]:
            return 0

        agent_loc = self.agent_location[agent]
        correct_point = self.delivery_correspondence[carried]
        delivery_loc = self.delivery_locations[correct_point]

        if agent_loc == delivery_loc:
            self.delivered[carried] = True
            self.agent_carrying[agent] = None
            return 1

        return 0

    # -------------------------------------------------------
    def _encode_carry(self, item):
        if item is None:
            return -1
        return int(item.split("_")[1])

    # -------------------------------------------------------
    def _get_observations(self):
        return {agent: self._single_observation(agent) for agent in self.agents}

    def _single_observation(self, agent):
        ax, ay = self.agent_location[agent]
        carried_item = self.agent_carrying[agent]

        R = self.obs_radius
        window = 2 * R + 1

        # Initialize with -1 everywhere (out of bounds)
        img = np.full((window, window, 4), -1.0, dtype=np.float32)

        def in_bounds(x, y):
            return 0 <= x < self.grid_size and 0 <= y < self.grid_size

        # ---------------------------------------------------------
        # Fill in-bounds tiles with 0 (empty tile)
        # ---------------------------------------------------------
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                wx, wy = ax + dx, ay + dy
                if in_bounds(wx, wy):
                    img[R + dx, R + dy, :] = 0.0

        # ---------------------------------------------------------
        # Channel 0: Ego agent
        # ---------------------------------------------------------
        img[R, R, 0] = 1.0

        # ---------------------------------------------------------
        # Channel 1: Other agents
        # ---------------------------------------------------------
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            if -R <= dx <= R and -R <= dy <= R and in_bounds(ox, oy):
                img[R + dx, R + dy, 1] = 1.0

        # ---------------------------------------------------------
        # Channel 2: Items (only undelivered)
        # ---------------------------------------------------------
        for item in self.items:
            if self.delivered[item]:
                continue
            ix, iy = self.item_locations[item]
            dx, dy = ix - ax, iy - ay
            if -R <= dx <= R and -R <= dy <= R and in_bounds(ix, iy):
                img[R + dx, R + dy, 2] = 1.0

        # ---------------------------------------------------------
        # Channel 3: Delivery point for the item being carried
        # ---------------------------------------------------------
        if carried_item is not None:
            # Lookup correct delivery point
            dp = self.delivery_correspondence[carried_item]
            tx, ty = self.delivery_locations[dp]

            dx, dy = tx - ax, ty - ay

            # Only mark if in the window
            if -R <= dx <= R and -R <= dy <= R and in_bounds(tx, ty):
                img[R + dx, R + dy, 3] = 1.0

        # ---------------------------------------------------------
        # Separate scalar indicating whether the agent is carrying
        # ---------------------------------------------------------
        carry_flag = np.array([1.0 if carried_item is not None else 0.0], dtype=np.float32)

        return {"image": img, "carry": carry_flag}
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    # -------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        self._init_pygame()

        # Handle window close / events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return

        # Background
        self._screen.fill((30, 30, 30))

        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                sx, sy = self._grid_to_screen(x, y)
                rect = pygame.Rect(
                    sx, sy,
                    self._cell_size, self._cell_size
                )
                pygame.draw.rect(self._screen, (60, 60, 60), rect, width=1)

        # Draw delivery points (as squares with border)
        for point in self.delivery_points:
            dx, dy = self.delivery_locations[point]
            sx, sy = self._grid_to_screen(dx, dy)
            rect = pygame.Rect(
                sx + 4, sy + 4,
                self._cell_size - 8, self._cell_size - 8
            )
            color = self.delivery_colors[point]
            pygame.draw.rect(self._screen, color, rect)
            pygame.draw.rect(self._screen, (255, 255, 255), rect, width=2)

        # Draw items (as circles)
        for item in self.items:
            if self.delivered[item]:
                continue  # don't show delivered items
            ix, iy = self.item_locations[item]
            sx, sy = self._grid_to_screen(ix, iy)
            center = (sx + self._cell_size // 2, sy + self._cell_size // 2)
            color = self.item_colors[item]
            pygame.draw.circle(self._screen, color, center, self._cell_size // 4)

        # Draw agents (as colored squares)
        agent_color = (0, 200, 255)     # cyan
        for agent in self.agents:
            ax, ay = self.agent_location[agent]
            sx, sy = self._grid_to_screen(ax, ay)
            rect = pygame.Rect(
                sx + 8, sy + 8,
                self._cell_size - 16, self._cell_size - 16
            )
            pygame.draw.rect(self._screen, agent_color, rect)

            # If carrying, outline white
            if self.agent_carrying[agent] is not None:
                pygame.draw.rect(self._screen, (255, 255, 255), rect, width=2)

        # Optional text overlay: timestep
        text = self._font.render(f"t = {self.timestep}", True, (220, 220, 220))
        self._screen.blit(text, (10, 10))

        pygame.display.flip()
        self._clock.tick(10)  # 10 FPS cap


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
        
    def _grid_to_screen(self, x, y):
        """
        Convert grid coords (x,y) with (0,0) at bottom-left
        into pygame pixel coords with (0,0) at top-left.
        """
        sx = self._margin + x * self._cell_size
        sy = self._margin + (self.grid_size - 1 - y) * self._cell_size
        return sx, sy
    
    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False


if __name__ == "__main__":
    env = PickupDelivery(
        grid_size=7,
        num_agents=3,
        num_items=4,
        num_delivery_points=2
    )

    parallel_api_test(env, num_cycles=10000)
    
    obs, _ = env.reset()

    for _ in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rew, done, trunc, info = env.step(actions)
        env.render()
        if all(done.values()) or all(trunc.values()):
            break

    env.close()