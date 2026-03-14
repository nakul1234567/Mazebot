import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

import numpy as np
import gymnasium as gym
from collections import deque

# ---------------- Reward memory ----------------
_LAST_OBS = None
_PREV_POS = None
_PREV_PREV_POS = None
_PREV_FRONTIER_DIST = None
_PREV_DANGER_DIST = None
_PREV_ENEMY_DIST = None
_PREV_STEPS_REMAINING = None
_RECENT_POSITIONS = deque(maxlen=6)


def _reset_reward_memory():
    global _LAST_OBS, _PREV_POS, _PREV_PREV_POS
    global _PREV_FRONTIER_DIST, _PREV_DANGER_DIST, _PREV_ENEMY_DIST
    global _PREV_STEPS_REMAINING, _RECENT_POSITIONS

    _LAST_OBS = None
    _PREV_POS = None
    _PREV_PREV_POS = None
    _PREV_FRONTIER_DIST = None
    _PREV_DANGER_DIST = None
    _PREV_ENEMY_DIST = None
    _PREV_STEPS_REMAINING = None
    _RECENT_POSITIONS = deque(maxlen=6)


def _min_manhattan_distance(mask: np.ndarray, row: int, col: int):
    """
    mask: 2D boolean array
    returns minimum Manhattan distance from (row, col) to any True cell, or None
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    dists = np.abs(coords[:, 0] - row) + np.abs(coords[:, 1] - col)
    return int(dists.min())

# ---- Local copies of the RGB colors used by env.py ----
# We define them here instead of importing from env.py to avoid circular imports.

BLACK = np.array([0, 0, 0], dtype=np.uint8)            # unexplored
WHITE = np.array([255, 255, 255], dtype=np.uint8)      # explored
BROWN = np.array([101, 67, 33], dtype=np.uint8)        # wall
GREY = np.array([160, 161, 161], dtype=np.uint8)       # agent
GREEN = np.array([31, 198, 0], dtype=np.uint8)         # enemy
RED = np.array([255, 0, 0], dtype=np.uint8)            # unexplored + danger
LIGHT_RED = np.array([255, 127, 127], dtype=np.uint8)  # explored + danger


def _color_mask(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return a boolean mask for cells whose RGB value matches `color`."""
    return np.all(grid == color, axis=2)


def _frontier_mask(unexplored_mask: np.ndarray, explored_mask: np.ndarray) -> np.ndarray:
    """
    Frontier = unexplored cells that are adjacent (4-neighbourhood)
    to at least one explored cell.
    """
    adjacent_to_explored = np.zeros_like(explored_mask, dtype=bool)

    # A cell is adjacent to an explored cell if one of its 4 neighbors is explored
    adjacent_to_explored[1:, :] |= explored_mask[:-1, :]   # explored above
    adjacent_to_explored[:-1, :] |= explored_mask[1:, :]   # explored below
    adjacent_to_explored[:, 1:] |= explored_mask[:, :-1]   # explored left
    adjacent_to_explored[:, :-1] |= explored_mask[:, 1:]   # explored right

    return unexplored_mask & adjacent_to_explored


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    7 x 10 x 10 float observation:
      0: walls
      1: unexplored cells
      2: explored cells
      3: enemy FOV / danger cells
      4: agent position
      5: enemy positions
      6: frontier cells
    """
    h, w = env.grid.shape[:2]
    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(7, h, w),
        dtype=np.float32
    )


def observation(grid: np.ndarray) -> np.ndarray:
    """
    Convert the RGB grid into a 7-channel binary tensor.
    """
    # Basic masks from colors
    wall_mask = _color_mask(grid, BROWN)
    black_mask = _color_mask(grid, BLACK)
    white_mask = _color_mask(grid, WHITE)
    grey_mask = _color_mask(grid, GREY)
    green_mask = _color_mask(grid, GREEN)
    red_mask = _color_mask(grid, RED)
    light_red_mask = _color_mask(grid, LIGHT_RED)

    # Channels
    walls = wall_mask

    # RED cells are still unexplored, just dangerous
    unexplored = black_mask | red_mask

    # Agent cell is already explored; LIGHT_RED is explored+danger
    explored = white_mask | light_red_mask | grey_mask

    # Danger/FOV cells only
    danger = red_mask | light_red_mask

    agent = grey_mask
    enemies = green_mask

    frontier = _frontier_mask(unexplored, explored)

    obs = np.stack(
        [
            walls.astype(np.float32),
            unexplored.astype(np.float32),
            explored.astype(np.float32),
            danger.astype(np.float32),
            agent.astype(np.float32),
            enemies.astype(np.float32),
            frontier.astype(np.float32),
        ],
        axis=0
    )
    global _LAST_OBS
    _LAST_OBS = obs

    return obs


def reward(info: dict) -> float:
    global _PREV_POS, _PREV_PREV_POS
    global _PREV_FRONTIER_DIST, _PREV_DANGER_DIST, _PREV_ENEMY_DIST
    global _PREV_STEPS_REMAINING, _RECENT_POSITIONS

    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # ---------------- Detect new episode ----------------
    # During an episode, steps_remaining only decreases.
    # If it suddenly increases, we are in a new episode.
    if _PREV_STEPS_REMAINING is None or steps_remaining > _PREV_STEPS_REMAINING:
        _PREV_POS = None
        _PREV_PREV_POS = None
        _PREV_FRONTIER_DIST = None
        _PREV_DANGER_DIST = None
        _PREV_ENEMY_DIST = None
        _RECENT_POSITIONS = deque(maxlen=6)

    # Convert flattened position to (row, col)
    row = agent_pos // 10
    col = agent_pos % 10

    r = 0.0

    # ---------------- 1) Small step cost ----------------
    # Encourages faster completion
    r -= 0.01

    # ---------------- 2) Main exploration reward ----------------
    if new_cell_covered:
        # Strong reward for actually expanding coverage
        # Slightly higher later in the episode
        progress_ratio = total_covered_cells / max(1, coverable_cells)
        r += 1.2 + 0.4 * progress_ratio
    else:
        # No new cell covered
        if _PREV_POS is not None and agent_pos == _PREV_POS:
            # likely STAY, wall hit, enemy hit, or blocked move
            r -= 0.18
        else:
            # moved, but only over already explored territory
            r -= 0.06

    # ---------------- 3) Frontier shaping ----------------
    # Uses channel 6 from your observation
    if _LAST_OBS is not None:
        frontier_mask = _LAST_OBS[6] > 0.5
        frontier_dist = _min_manhattan_distance(frontier_mask, row, col)

        if frontier_dist is not None:
            # Reward moving closer to frontier, penalize moving away
            if _PREV_FRONTIER_DIST is not None:
                delta = _PREV_FRONTIER_DIST - frontier_dist
                r += 0.05 * np.clip(delta, -2, 2)

            # Small bonus for staying adjacent to frontier
            if frontier_dist == 1:
                r += 0.04

            _PREV_FRONTIER_DIST = frontier_dist
        else:
            _PREV_FRONTIER_DIST = None

    # ---------------- 4) Safety shaping ----------------
    # Uses danger cells (channel 3) and enemy positions (channel 5)
    if _LAST_OBS is not None:
        danger_mask = _LAST_OBS[3] > 0.5
        enemy_mask = _LAST_OBS[5] > 0.5

        danger_dist = _min_manhattan_distance(danger_mask, row, col)
        enemy_dist = _min_manhattan_distance(enemy_mask, row, col)

        # Penalize being too close to danger zones
        if danger_dist is not None:
            if danger_dist <= 1:
                r -= 0.18
            elif danger_dist == 2:
                r -= 0.08

            # Small reward for moving away from danger
            if _PREV_DANGER_DIST is not None:
                delta = danger_dist - _PREV_DANGER_DIST
                r += 0.03 * np.clip(delta, -2, 2)

            _PREV_DANGER_DIST = danger_dist
        else:
            _PREV_DANGER_DIST = None

        # Penalize hugging enemies directly
        if enemy_dist is not None:
            if enemy_dist <= 1:
                r -= 0.15
            elif enemy_dist == 2:
                r -= 0.05

            _PREV_ENEMY_DIST = enemy_dist
        else:
            _PREV_ENEMY_DIST = None

    # ---------------- 5) Anti-loop / anti-oscillation ----------------
    if _PREV_PREV_POS is not None and agent_pos == _PREV_PREV_POS and not new_cell_covered:
        # A-B-A oscillation
        r -= 0.12

    if agent_pos in _RECENT_POSITIONS and not new_cell_covered:
        r -= 0.05

    # ---------------- 6) Terminal rewards ----------------
    covered_ratio = total_covered_cells / max(1, coverable_cells)

    if cells_remaining == 0:
        # Big success bonus, plus a little extra for finishing early
        r += 12.0 + 3.0 * (steps_remaining / 500.0)

    elif game_over:
        # Strong penalty for getting seen
        r -= 12.0

    elif steps_remaining <= 0 and cells_remaining > 0:
        # Timeout penalty depends on how incomplete the map is
        r -= 4.0 + 4.0 * (1.0 - covered_ratio)

    # ---------------- Update memory ----------------
    _RECENT_POSITIONS.append(agent_pos)
    _PREV_PREV_POS = _PREV_POS
    _PREV_POS = agent_pos
    _PREV_STEPS_REMAINING = steps_remaining

    return float(r)