from typing import Dict
import random
import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class Foraging(ParallelEnv):
    """
    Unified Multi-Agent Foraging Environment.

    Observation modes:
        - "vector": flat structured vector (global coordinates)
        - "window": local window (image-like)
        - "knn":    K-nearest neighbor encoded vector

    General Task: Cooperative Multi-Agent Foraging
    """

    metadata = {
        "render_modes": ["human"],
        "name": "Foraging-v0",
    }

    def __init__(
        self,
        grid_size=7,
        num_agents=2,
        num_items=3,
        obs_mode="vector",      # "vector", "window", or "knn"
        obs_radius=3,           # used only for window mode
        k_agents=2,             # used only for knn mode
        k_items=2,
    ):

        assert obs_mode in ("vector", "window", "knn")
        self.obs_mode = obs_mode

        self.grid_size = grid_size
        self.n_agents = num_agents
        self.n_items = num_items
        self.obs_radius = obs_radius
        self.k_agents = k_agents
        self.k_items = k_items

        self.max_steps = 200
        self.timestep = 0

        # Agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_location = {a: (0, 0) for a in self.agents}

        # Items
        self.items = [f"item_{i}" for i in range(num_items)]
        self.item_locations = {item: (0, 0) for item in self.items}
        self.collected = {item: False for item in self.items}

        # Action space
        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # Observation spaces depend on mode
        self.observation_spaces = {
            agent: self._build_observation_space()
            for agent in self.possible_agents
        }

        # Rendering
        self.render_mode = "human"
        self._pygame_initialized = False
        self._cell_size = 64
        self._margin = 40
        self._screen = None
        self._clock = None
        self._font = None


    # ============================================================
    # Spaces
    # ============================================================
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _build_observation_space(self):
        if self.obs_mode == "vector":
            size = 2 + (self.n_agents - 1) * 2 + self.n_items * 2
            return spaces.Box(low=-1, high=self.grid_size, shape=(size,), dtype=np.float32)

        elif self.obs_mode == "window":
            w = 2 * self.obs_radius + 1
            return spaces.Box(low=-1, high=1, shape=(w, w, 3), dtype=np.float32)

        elif self.obs_mode == "knn":
            size = (self.k_agents + self.k_items) * 3
            return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)


    # ============================================================
    # Reset
    # ============================================================
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = self.possible_agents[:]
        self.collected = {item: False for item in self.items}

        needed = self.n_agents + self.n_items
        positions = set()

        while len(positions) < needed:
            positions.add((random.randint(0, self.grid_size - 1),
                           random.randint(0, self.grid_size - 1)))

        positions = list(positions)
        idx = 0

        # Items
        for item in self.items:
            self.item_locations[item] = positions[idx]
            idx += 1

        # Agents
        for agent in self.agents:
            self.agent_location[agent] = positions[idx]
            idx += 1

        return self._get_observations(), {a: {} for a in self.agents}


    # ============================================================
    # Step
    # ============================================================
    def step(self, actions):
        self.timestep += 1
        rewards = {agent: -0.01 for agent in self.agents}

        for agent, action in actions.items():

            if action in [0, 1, 2, 3]:
                self.agent_location[agent] = self._move(self.agent_location[agent], action)

            elif action == 4:
                loc = self.agent_location[agent]
                for item, it_loc in self.item_locations.items():
                    if not self.collected[item] and it_loc == loc:
                        self.collected[item] = True
                        rewards[agent] += 1.0
                        break

        done = all(self.collected.values())
        truncated = self.timestep >= self.max_steps

        dones = {a: done or truncated for a in self.agents}
        truncs = {a: truncated for a in self.agents}

        if done or truncated:
            self.agents = []

        return (
            self._get_observations(),
            rewards,
            dones,
            truncs,
            {a: {} for a in dones},
        )


    # ============================================================
    # Movement
    # ============================================================
    def _move(self, loc, action):
        x, y = loc
        if action == 0:
            y = min(self.grid_size - 1, y + 1)
        elif action == 1:
            y = max(0, y - 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.grid_size - 1, x + 1)
        return (x, y)


    # ============================================================
    # Observation dispatcher
    # ============================================================
    def _get_observations(self):
        return {a: self._single_obs(a) for a in self.agents}

    def _single_obs(self, agent):
        if self.obs_mode == "vector":
            return self._obs_vector(agent)
        elif self.obs_mode == "window":
            return self._obs_window(agent)
        elif self.obs_mode == "knn":
            return self._obs_knn(agent)


    # ============================================================
    # Vector Observation
    # ============================================================
    def _obs_vector(self, agent):
        ax, ay = self.agent_location[agent]
        obs = [ax, ay]

        # others
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            obs.extend([ox, oy])

        # items
        for item in self.items:
            if self.collected[item]:
                obs.extend([-1, -1])
            else:
                ix, iy = self.item_locations[item]
                obs.extend([ix, iy])

        return np.array(obs, dtype=np.float32)


    # ============================================================
    # WINDOW Observation (formerly area)
    # ============================================================
    def _obs_window(self, agent):
        ax, ay = self.agent_location[agent]
        R = self.obs_radius
        W = 2 * R + 1

        obs = np.full((W, W, 3), -1.0, dtype=np.float32)

        def in_bounds(x, y):
            return 0 <= x < self.grid_size and 0 <= y < self.grid_size

        # empty cells
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                wx, wy = ax + dx, ay + dy
                if in_bounds(wx, wy):
                    obs[R + dx, R + dy, :] = 0.0

        # ego
        obs[R, R, 0] = 1.0

        # other agents
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            if -R <= dx <= R and -R <= dy <= R and in_bounds(ox, oy):
                obs[R + dx, R + dy, 1] = 1.0

        # items
        for item in self.items:
            if self.collected[item]:
                continue
            ix, iy = self.item_locations[item]
            dx, dy = ix - ax, iy - ay
            if -R <= dx <= R and -R <= dy <= R and in_bounds(ix, iy):
                obs[R + dx, R + dy, 2] = 1.0

        return obs


    # ============================================================
    # KNN Observation
    # ============================================================
    def _obs_knn(self, agent):
        ax, ay = self.agent_location[agent]

        # agents
        others = []
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            dist = abs(dx) + abs(dy)
            others.append((dist, dx, dy))
        others.sort(key=lambda x: x[0])

        # items
        items = []
        for item in self.items:
            if self.collected[item]:
                continue
            ix, iy = self.item_locations[item]
            dx, dy = ix - ax, iy - ay
            dist = abs(dx) + abs(dy)
            items.append((dist, dx, dy))
        items.sort(key=lambda x: x[0])

        result = []

        # nearest agents
        for i in range(self.k_agents):
            if i < len(others):
                _, dx, dy = others[i]
                result.extend([dx, dy, 1.0])
            else:
                result.extend([0.0, 0.0, 0.0])

        # nearest items
        for i in range(self.k_items):
            if i < len(items):
                _, dx, dy = items[i]
                result.extend([dx, dy, 1.0])
            else:
                result.extend([0.0, 0.0, 0.0])

        return np.array(result, dtype=np.float32)


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

        # grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                sx, sy = self._grid_to_screen(x, y)
                pygame.draw.rect(
                    self._screen,
                    (60, 60, 60),
                    pygame.Rect(sx, sy, self._cell_size, self._cell_size),
                    1,
                )

        # items
        for item in self.items:
            if self.collected[item]:
                continue
            ix, iy = self.item_locations[item]
            sx, sy = self._grid_to_screen(ix, iy)
            pygame.draw.circle(
                self._screen,
                (255, 200, 0),
                (sx + self._cell_size // 2, sy + self._cell_size // 2),
                self._cell_size // 4,
            )

        # agents
        for agent in self.agents:
            ax, ay = self.agent_location[agent]
            sx, sy = self._grid_to_screen(ax, ay)
            pygame.draw.rect(
                self._screen,
                (0, 200, 255),
                pygame.Rect(sx + 8, sy + 8, self._cell_size - 16, self._cell_size - 16),
            )

        text = self._font.render(f"t={self.timestep}", True, (255, 255, 255))
        self._screen.blit(text, (10, 10))

        pygame.display.flip()
        self._clock.tick(10)

    def _init_pygame(self):
        if self._pygame_initialized:
            return
        pygame.init()

        width = self.grid_size * self._cell_size + 2 * self._margin
        height = self.grid_size * self._cell_size + 2 * self._margin
        self._screen = pygame.display.set_mode((width, height))

        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 18)
        self._pygame_initialized = True

    def _grid_to_screen(self, x, y):
        return (
            self._margin + x * self._cell_size,
            self._margin + (self.grid_size - 1 - y) * self._cell_size,
        )

    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False


if __name__ == "__main__":
    # Example usage of the Foraging environment
    env = Foraging(grid_size=10, num_agents=4, num_items=4, obs_mode="vector")
    obs, info = env.reset()
    done = {a: False for a in env.agents}

    while not all(done.values()):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, done, trunc, info = env.step(actions)
        env.render()

    env.close()