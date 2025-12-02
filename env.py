from typing import Dict
import random
import numpy as np
from pettingzoo.utils import ParallelEnv
from pettingzoo.test import api_test, parallel_api_test
from gymnasium import spaces
import pygame


class PickupOnly(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "PickupOnly-v0",
    }

    def __init__(self, grid_size=7, num_agents=2, num_items=3):
        self.grid_size = grid_size
        self.n_agents = num_agents
        self.n_items = num_items

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

        # Actions: 0 up, 1 down, 2 left, 3 right, 4 grab
        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # Observation size:
        # self_loc(2) + other_agents(2 each: x,y) + items(2 each)
        obs_size = 2 + (num_agents - 1) * 2 + num_items * 2

        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=grid_size, shape=(obs_size,), dtype=np.float32)
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

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = self.possible_agents[:]
        self.collected = {item: False for item in self.items}

        # Sample unique positions
        needed = self.n_agents + self.n_items
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

        # Place agents
        for agent in self.agents:
            self.agent_location[agent] = positions[idx]
            idx += 1

        return self._get_observations(), {a: {} for a in self.agents}

    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------
    def step(self, actions):
        self.timestep += 1
        rewards = {agent: -0.01 for agent in self.agents}  # step penalty

        # Execute actions
        for agent, action in actions.items():

            # Movement
            if action in [0, 1, 2, 3]:
                old_loc = self.agent_location[agent]
                new_loc = self._move(old_loc, action)
                self.agent_location[agent] = new_loc


            # Grab → collect item if on item tile
            elif action == 4:
                loc = self.agent_location[agent]
                for item, i_loc in self.item_locations.items():
                    if not self.collected[item] and i_loc == loc:
                        self.collected[item] = True
                        rewards[agent] += 1.0
                        break

        # Termination
        done = all(self.collected.values())
        truncated = self.timestep >= self.max_steps

        dones = {a: done or truncated for a in self.agents}
        truncations = {a: truncated for a in self.agents}

        if done or truncated:
            self.agents = []

        return (
            self._get_observations(),
            rewards,
            dones,
            truncations,
            {a: {} for a in dones},
        )

    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def _move(self, loc, action):
        x, y = loc
        if action == 0:  # up
            y = min(self.grid_size - 1, y + 1)
        elif action == 1:  # down
            y = max(0, y - 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size - 1, x + 1)
        return (x, y)

    # ------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------
    def _get_observations(self):
        return {agent: self._single_obs(agent) for agent in self.agents}

    def _single_obs(self, agent):
        obs = []

        # self
        ax, ay = self.agent_location[agent]
        obs.extend([ax, ay])

        # other agents
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            obs.extend([ox, oy, 0])  # no carry

        # items
        for item in self.items:
            if self.collected[item]:
                obs.extend([-1, -1])
            else:
                ix, iy = self.item_locations[item]
                obs.extend([ix, iy])

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
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
                    self._screen, (60, 60, 60),
                    pygame.Rect(sx, sy, self._cell_size, self._cell_size), 1
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
                self._cell_size // 4
            )

        # agents
        for agent in self.agents:
            ax, ay = self.agent_location[agent]
            sx, sy = self._grid_to_screen(ax, ay)
            pygame.draw.rect(
                self._screen,
                (0, 200, 255),
                pygame.Rect(sx + 8, sy + 8, self._cell_size - 16, self._cell_size - 16)
            )

        # timestep
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

    # convert grid to screen
    def _grid_to_screen(self, x, y):
        return (
            self._margin + x * self._cell_size,
            self._margin + (self.grid_size - 1 - y) * self._cell_size,
        )

    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False


# Demo
if __name__ == "__main__":
    env = PickupOnly(grid_size=7, num_agents=3, num_items=4)
    parallel_api_test(env, num_cycles=10000)
    
    obs, _ = env.reset()
    for _ in range(200):
        acts = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, done, trunc, info = env.step(acts)
        env.render()
        if all(done.values()):
            break
    env.close()
