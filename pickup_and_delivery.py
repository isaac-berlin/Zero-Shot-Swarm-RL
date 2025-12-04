import random
from typing import Dict, List
import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class PickupDelivery(ParallelEnv):
    """
    Unified Pickup & Delivery environment supporting:
        obs_mode="vector"   → MLP-friendly flat observation
        obs_mode="window"   → Egocentric (CNN) observation
        obs_mode="knn"      → kNN feature-vector observation
    """

    metadata = {"render_modes": ["human"], "name": "PickupDelivery"}

    def __init__(
        self,
        grid_size=5,
        num_agents=2,
        num_items=3,
        num_delivery_points=1,
        obs_mode="vector",        # "vector", "window", or "knn"
        obs_radius=3,             # window size for CNN obs
        k_nearest_agents=2,       # kNN parameters
        k_nearest_items=3,
    ):

        assert obs_mode in ("vector", "window", "knn")
        self.obs_mode = obs_mode

        self.grid_size = grid_size
        self.n_agents = num_agents
        self.n_items = num_items
        self.n_delivery_points = num_delivery_points

        self.obs_radius = obs_radius
        self.k_nearest_agents = k_nearest_agents
        self.k_nearest_items = k_nearest_items

        self.max_steps = 200
        self.timestep = 0

        # ------------- Items & Deliveries -------------
        self.items = [f"item_{i}" for i in range(num_items)]
        self.item_locations = {item: (0, 0) for item in self.items}
        self.delivered = {item: False for item in self.items}

        self.delivery_points = [f"delivery_{i}" for i in range(num_delivery_points)]
        self.delivery_locations = {p: (0, 0) for p in self.delivery_points}
        self.delivery_correspondence = {item: None for item in self.items}

        # ------------- Agents -------------
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.agent_location = {a: (0, 0) for a in self.agents}
        self.agent_carrying = {a: None for a in self.agents}

        # Movement + Interaction
        # 0 up, 1 down, 2 left, 3 right, 4 interact (pick/drop/deliver)
        self.action_spaces = {
            a: spaces.Discrete(5) for a in self.possible_agents
        }

        # -------------------------
        # Build observation spaces
        # -------------------------
        self.observation_spaces = {
            a: self._build_obs_space()
            for a in self.possible_agents
        }

        # Rendering
        self.render_mode = "human"
        self._pygame_initialized = False
        self._cell_size = 64
        self._margin = 40
        self._screen = None
        self._clock = None
        self._font = None
        self.item_colors = {}
        self.delivery_colors = {}

    # =====================================================================
    # Observation spaces
    # =====================================================================
    def _build_obs_space(self):
        if self.obs_mode == "vector":
            # From env_PnD.py
            return spaces.Box(
                low=-self.grid_size,
                high=self.grid_size,
                shape=(
                    2                       # self loc
                    + 1                     # self carry idx
                    + (self.n_agents - 1)*3 # others loc+carry
                    + self.n_items*2        # item loc
                    + self.n_delivery_points*2, # delivery loc
                ),
                dtype=np.float32,
            )

        elif self.obs_mode == "window":
            # (H,W,4) as in env_PnD_area.py
            w = 2*self.obs_radius + 1
            return spaces.Dict({
                "image": spaces.Box(
                    low=-1, high=1,
                    shape=(w, w, 4),
                    dtype=np.float32
                ),
                "carry": spaces.Box(
                    low=0, high=1,
                    shape=(1,),
                    dtype=np.float32
                )
            })

        elif self.obs_mode == "knn":
            # From env_PnD_knn.py
            self_dim = 3  # carry flag + dx target + dy target
            obs_dim = (
                self_dim
                + self.k_nearest_agents*2
                + self.k_nearest_items*2
            )
            return spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )

    # =====================================================================
    # Reset
    # =====================================================================
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = self.possible_agents[:]
        self.delivered = {i: False for i in self.items}
        self.agent_carrying = {a: None for a in self.agents}

        # Assign item → delivery mapping
        for item in self.items:
            self.delivery_correspondence[item] = random.choice(self.delivery_points)

        # Colors for visualization
        base_colors = [
            (255, 80, 80),
            (255, 200, 0),
            (80, 255, 120),
            (200, 80, 255),
            (255, 150, 80),
        ]
        for i, p in enumerate(self.delivery_points):
            self.delivery_colors[p] = base_colors[i % len(base_colors)]

        for item in self.items:
            dp = self.delivery_correspondence[item]
            self.item_colors[item] = self.delivery_colors[dp]

        # Unique random placement
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
            self.item_locations[item] = positions[idx]; idx += 1

        # Delivery points
        for p in self.delivery_points:
            self.delivery_locations[p] = positions[idx]; idx += 1

        # Agents
        for a in self.agents:
            self.agent_location[a] = positions[idx]; idx += 1

        return self._get_observations(), {a: {} for a in self.agents}

    # =====================================================================
    # Step
    # =====================================================================
    def step(self, actions):
        self.timestep += 1
        rewards = {a: -0.01 for a in self.agents}

        for agent, action in actions.items():
            if action in [0,1,2,3]:
                self.agent_location[agent] = self._move(self.agent_location[agent], action)

            elif action == 4:
                carried = self.agent_carrying[agent]

                if carried is None:
                    if self._attempt_pickup(agent):
                        rewards[agent] += 0.5
                else:
                    if self._attempt_delivery(agent):
                        rewards[agent] += 5.0
                    else:
                        if self._attempt_drop(agent):
                            rewards[agent] -= 0.5

        done = all(self.delivered.values())
        truncated = self.timestep >= self.max_steps

        dones = {a: done or truncated for a in self.agents}
        trunc = {a: truncated for a in self.agents}

        if done or truncated:
            self.agents = []

        return self._get_observations(), rewards, dones, trunc, {a:{} for a in dones}

    # =====================================================================
    # Movement + Interaction
    # =====================================================================
    def _move(self, loc, action):
        x, y = loc
        if action == 0: y = min(self.grid_size-1, y+1)
        elif action == 1: y = max(0, y-1)
        elif action == 2: x = max(0, x-1)
        elif action == 3: x = min(self.grid_size-1, x+1)
        return (x, y)

    def _attempt_pickup(self, agent):
        if self.agent_carrying[agent] is not None:
            return False
        ax, ay = self.agent_location[agent]
        for item,(ix,iy) in self.item_locations.items():
            if not self.delivered[item] and (ax,ay)==(ix,iy):
                self.agent_carrying[agent] = item
                return True
        return False

    def _attempt_drop(self, agent):
        item = self.agent_carrying[agent]
        if item is None: return False
        self.item_locations[item] = self.agent_location[agent]
        self.agent_carrying[agent] = None
        return True

    def _attempt_delivery(self, agent):
        item = self.agent_carrying[agent]
        if item is None or self.delivered[item]:
            return False

        ax, ay = self.agent_location[agent]
        dp = self.delivery_correspondence[item]
        dx, dy = self.delivery_locations[dp]

        if (ax,ay)==(dx,dy):
            self.delivered[item]=True
            self.agent_carrying[agent]=None
            return True
        return False

    # =====================================================================
    # Observation dispatch
    # =====================================================================
    def _get_observations(self):
        return {a: self._single_obs(a) for a in self.agents}

    def _single_obs(self, agent):
        if self.obs_mode == "vector":
            return self._obs_vector(agent)
        elif self.obs_mode == "window":
            return self._obs_window(agent)
        else:
            return self._obs_knn(agent)

    # =====================================================================
    # Vector observation (from env_PnD.py)
    # =====================================================================
    def _obs_vector(self, agent):
        ax, ay = self.agent_location[agent]

        out = []
        # self loc + carry idx
        out.extend([ax, ay])
        out.append(self._encode_carry(self.agent_carrying[agent]))

        # others
        for other in self.agents:
            if other == agent: continue
            ox, oy = self.agent_location[other]
            out.extend([ox, oy, self._encode_carry(self.agent_carrying[other])])

        # items
        for item in self.items:
            ix, iy = self.item_locations[item]
            out.extend([ix, iy])

        # deliveries
        for p in self.delivery_points:
            dx, dy = self.delivery_locations[p]
            out.extend([dx, dy])

        return np.array(out, np.float32)

    # =====================================================================
    # Window observation (from env_PnD_area.py)
    # =====================================================================
    def _obs_window(self, agent):
        ax, ay = self.agent_location[agent]
        carried = self.agent_carrying[agent]

        R = self.obs_radius
        W = 2*R + 1

        img = np.full((W,W,4), -1.0, np.float32)

        def in_bounds(x,y):
            return 0 <= x < self.grid_size and 0 <= y < self.grid_size

        # blank tiles
        for dx in range(-R,R+1):
            for dy in range(-R,R+1):
                wx, wy = ax+dx, ay+dy
                if in_bounds(wx,wy):
                    img[R+dx, R+dy, :] = 0.0

        # Ego
        img[R,R,0] = 1.0

        # Other agents
        for other in self.agents:
            if other==agent: continue
            ox,oy = self.agent_location[other]
            dx,dy = ox-ax, oy-ay
            if -R<=dx<=R and -R<=dy<=R and in_bounds(ox,oy):
                img[R+dx, R+dy, 1] = 1.0

        # Items
        for item in self.items:
            if self.delivered[item]: continue
            ix,iy = self.item_locations[item]
            dx,dy = ix-ax, iy-ay
            if -R<=dx<=R and -R<=dy<=R and in_bounds(ix,iy):
                img[R+dx,R+dy,2] = 1.0

        # Delivery point of carried item
        if carried is not None:
            dp = self.delivery_correspondence[carried]
            tx,ty = self.delivery_locations[dp]
            dx,dy = tx-ax, ty-ay
            if -R<=dx<=R and -R<=dy<=R and in_bounds(tx,ty):
                img[R+dx,R+dy,3] = 1.0

        carry_flag = np.array([1.0 if carried else 0.0], np.float32)
        return {"image": img, "carry": carry_flag}

    # =====================================================================
    # KNN observation (from env_PnD_knn.py)
    # =====================================================================
    def _obs_knn(self, agent):
        ax, ay = self.agent_location[agent]
        carried = self.agent_carrying[agent]

        feat = []
        # Carry flag + dx target + dy target
        feat.append(1.0 if carried else 0.0)

        if carried is None:
            feat.extend([0.0,0.0])
        else:
            dp = self.delivery_correspondence[carried]
            tx,ty = self.delivery_locations[dp]
            feat.extend([tx-ax, ty-ay])

        # nearest agents
        others = [
            (other, self.agent_location[other])
            for other in self.agents if other!=agent
        ]
        others.sort(key=lambda x: self._euclid(self.agent_location[agent], x[1]))

        for i in range(self.k_nearest_agents):
            if i < len(others):
                _, (ox,oy) = others[i]
                feat.extend([ox-ax, oy-ay])
            else:
                feat.extend([0.0,0.0])

        # nearest items
        items = [
            (item, loc)
            for item,loc in self.item_locations.items()
            if not self.delivered[item]
        ]
        items.sort(key=lambda x: self._euclid(self.agent_location[agent], x[1]))

        for i in range(self.k_nearest_items):
            if i < len(items):
                _, (ix,iy) = items[i]
                feat.extend([ix-ax, iy-ay])
            else:
                feat.extend([0.0,0.0])

        return np.array(feat, np.float32)

    def _encode_carry(self, item):
        if item is None: return -1
        return int(item.split("_")[1])

    def _euclid(self, a,b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # =====================================================================
    # Rendering
    # =====================================================================
    def render(self):
        if self.render_mode!="human": return
        self._init_pygame()

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit(); self._pygame_initialized=False; return

        self._screen.fill((30,30,30))

        # GRID
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                sx,sy = self._grid_to_screen(x,y)
                pygame.draw.rect(
                    self._screen,(60,60,60),
                    pygame.Rect(sx,sy,self._cell_size,self._cell_size),
                    1
                )

        # Delivery points
        for p in self.delivery_points:
            dx,dy = self.delivery_locations[p]
            sx,sy = self._grid_to_screen(dx,dy)
            rect = pygame.Rect(
                sx+4, sy+4, self._cell_size-8, self._cell_size-8
            )
            pygame.draw.rect(self._screen, self.delivery_colors[p], rect)
            pygame.draw.rect(self._screen, (255,255,255), rect, 2)

        # Items
        for item in self.items:
            if self.delivered[item]: continue
            ix,iy = self.item_locations[item]
            sx,sy = self._grid_to_screen(ix,iy)
            pygame.draw.circle(
                self._screen, self.item_colors[item],
                (sx+self._cell_size//2, sy+self._cell_size//2),
                self._cell_size//4
            )

        # Agents
        for a in self.agents:
            ax,ay = self.agent_location[a]
            sx,sy = self._grid_to_screen(ax,ay)
            rect = pygame.Rect(sx+8, sy+8, self._cell_size-16, self._cell_size-16)
            pygame.draw.rect(self._screen, (0,200,255), rect)
            if self.agent_carrying[a]:
                pygame.draw.rect(self._screen,(255,255,255),rect,2)

        # timestep text
        text = self._font.render(f"t = {self.timestep}", True, (220, 220, 220))
        self._screen.blit(text, (10, 10))

        pygame.display.flip()
        self._clock.tick(10)


    # =====================================================================
    # Pygame init / teardown
    # =====================================================================
    def _init_pygame(self):
        if self._pygame_initialized:
            return

        pygame.init()
        pygame.display.set_caption("Unified Pickup & Delivery")

        width = self.grid_size * self._cell_size + 2 * self._margin
        height = self.grid_size * self._cell_size + 2 * self._margin

        self._screen = pygame.display.set_mode((width, height))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 18)

        self._pygame_initialized = True


    def _grid_to_screen(self, x, y):
        """
        Convert grid coords (x,y) with (0,0) at bottom-left
        into pygame pixel coords (0,0) at top-left.
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
        num_delivery_points=2,
        obs_mode="vector",  # "vector", "window", "knn"
    )

    obs, info = env.reset()
    done_flags = {a: False for a in env.possible_agents}

    while not all(done_flags.values()):
        env.render()

        actions = {}
        for a in env.possible_agents:
            actions[a] = env.action_spaces[a].sample()

        obs, rewards, dones, truncs, infos = env.step(actions)

        done_flags = {a: dones.get(a, True) or truncs.get(a, True) for a in env.possible_agents}

    env.close()