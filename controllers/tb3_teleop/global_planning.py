"""Costmap-aware global A* planner for TECH WEEK AMR."""

import heapq
import math

from global_costmap import INSCRIBED_COST, MAX_INFLATION_COST


SQRT2 = math.sqrt(2.0)
EPS = 1e-9

# Small enough that distance still matters, but large inflation cost is avoided
# when an only-slightly-longer safer route exists.
INFLATION_COST_WEIGHT = 0.35

# A smoothing shortcut may be at most this much more expensive than the raw
# A* subsection it replaces.  This prevents smoothing from cutting too close
# to an obstacle and undoing the costmap-aware route choice.
SMOOTH_COST_FACTOR = 1.05

MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
)


def _in_bounds(shape, cell):
    row, col = cell
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _is_blocked(costmap, cell):
    row, col = cell
    return int(costmap[row, col]) >= int(INSCRIBED_COST)


def _normalized_inflation_cost(costmap, cell):
    cost = int(costmap[cell])
    if cost <= 0 or cost >= int(INSCRIBED_COST):
        return 0.0
    return cost / MAX_INFLATION_COST


def _step_cost(costmap, nxt, geometric_cost):
    penalty = INFLATION_COST_WEIGHT * _normalized_inflation_cost(costmap, nxt)
    return geometric_cost * (1.0 + penalty)


def _heuristic(a, b):
    """Octile distance for 8-connected movement."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diagonal = min(dr, dc)
    straight = max(dr, dc) - diagonal
    return diagonal * SQRT2 + straight


def _can_step(costmap, current, nxt):
    if not _in_bounds(costmap.shape, nxt):
        return False

    if _is_blocked(costmap, nxt):
        return False

    dr = nxt[0] - current[0]
    dc = nxt[1] - current[1]

    # Prevent diagonal corner cutting.
    if dr != 0 and dc != 0:
        side1 = (current[0] + dr, current[1])
        side2 = (current[0], current[1] + dc)

        if _is_blocked(costmap, side1) or _is_blocked(costmap, side2):
            return False

    return True


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _astar(costmap, start, goals):
    goal_set = set(goals)

    if not goal_set:
        return None

    if start in goal_set:
        return [start]

    def heuristic_to_goals(cell):
        return min(_heuristic(cell, goal) for goal in goal_set)

    came_from = {}
    g_score = {start: 0.0}
    open_heap = []
    closed = set()
    push_id = 0

    h0 = heuristic_to_goals(start)
    heapq.heappush(open_heap, (h0, int(costmap[start]), h0, push_id, 0.0, start))

    while open_heap:
        _, _, _, _, current_g, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current_g > g_score.get(current, math.inf) + EPS:
            continue

        if current in goal_set:
            return _reconstruct_path(came_from, current)

        closed.add(current)

        for dr, dc, geometric_cost in MOVES:
            nxt = (current[0] + dr, current[1] + dc)

            if not _can_step(costmap, current, nxt):
                continue

            tentative_g = current_g + _step_cost(costmap, nxt, geometric_cost)

            if tentative_g + EPS >= g_score.get(nxt, math.inf):
                continue

            came_from[nxt] = current
            g_score[nxt] = tentative_g
            h_score = heuristic_to_goals(nxt)
            f_score = tentative_g + h_score
            push_id += 1

            # If f is equal, lower costmap value is preferred.
            heapq.heappush(
                open_heap,
                (f_score, int(costmap[nxt]), h_score, push_id, tentative_g, nxt),
            )

    return None


def _ring_candidates(costmap, center, radius):
    center_row, center_col = center
    candidates = []

    for row in range(center_row - radius, center_row + radius + 1):
        for col in range(center_col - radius, center_col + radius + 1):
            if max(abs(row - center_row), abs(col - center_col)) != radius:
                continue

            cell = (row, col)
            if not _in_bounds(costmap.shape, cell):
                continue
            if _is_blocked(costmap, cell):
                continue

            candidates.append(cell)

    return candidates


def _plan_to_goal_or_approach(costmap, start, target):
    if not _is_blocked(costmap, target):
        return _astar(costmap, start, [target])

    max_radius = max(costmap.shape)

    for radius in range(1, max_radius + 1):
        candidates = _ring_candidates(costmap, target, radius)
        if not candidates:
            continue

        path = _astar(costmap, start, candidates)
        if path is not None:
            return path

    return None


def _bresenham_cells(start, end):
    row0, col0 = start
    row1, col1 = end
    dcol = abs(col1 - col0)
    drow = abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1
    error = dcol - drow
    cells = []

    while True:
        cells.append((row0, col0))
        if row0 == row1 and col0 == col1:
            break

        error2 = 2 * error
        if error2 > -drow:
            error -= drow
            col0 += step_col
        if error2 < dcol:
            error += dcol
            row0 += step_row

    return cells


def _line_of_sight(costmap, start, end):
    cells = _bresenham_cells(start, end)
    previous = cells[0]

    for cell in cells:
        if not _in_bounds(costmap.shape, cell) or _is_blocked(costmap, cell):
            return False

        dr = cell[0] - previous[0]
        dc = cell[1] - previous[1]

        if dr != 0 and dc != 0:
            side1 = (previous[0] + dr, previous[1])
            side2 = (previous[0], previous[1] + dc)
            if _is_blocked(costmap, side1) or _is_blocked(costmap, side2):
                return False

        previous = cell

    return True


def _cells_cost(costmap, cells):
    total = 0.0

    for previous, current in zip(cells, cells[1:]):
        dr = abs(current[0] - previous[0])
        dc = abs(current[1] - previous[1])
        geometric_cost = SQRT2 if dr and dc else 1.0
        total += _step_cost(costmap, current, geometric_cost)

    return total


def _smooth_path(costmap, path):
    """Remove unnecessary waypoints without undoing the costmap preference."""
    if path is None or len(path) <= 2:
        return path

    smoothed = [path[0]]
    anchor = 0

    while anchor < len(path) - 1:
        next_index = anchor + 1

        for candidate in range(len(path) - 1, anchor, -1):
            if not _line_of_sight(costmap, path[anchor], path[candidate]):
                continue

            direct_cells = _bresenham_cells(path[anchor], path[candidate])
            direct_cost = _cells_cost(costmap, direct_cells)
            original_cost = _cells_cost(costmap, path[anchor:candidate + 1])

            if direct_cost <= original_cost * SMOOTH_COST_FACTOR + EPS:
                next_index = candidate
                break

        smoothed.append(path[next_index])
        anchor = next_index

    return smoothed


def plan(grid, costmap, pose, goal):
    """Return world-coordinate waypoints from the next point to the destination."""
    if goal is None:
        return None

    if grid.data.shape != costmap.shape:
        return None

    start = grid.world_to_grid(pose.x, pose.y)
    target = grid.world_to_grid(goal[0], goal[1])

    if start is None or target is None:
        return None

    start = tuple(start)
    target = tuple(target)

    if not _in_bounds(costmap.shape, start) or not _in_bounds(costmap.shape, target):
        return None

    if start == target:
        return []

    planning_costmap = costmap.copy()
    planning_costmap[start] = 0

    cell_path = _plan_to_goal_or_approach(planning_costmap, start, target)

    if cell_path is None:
        return None

    cell_path = _smooth_path(planning_costmap, cell_path)

    world_path = []
    for row, col in cell_path[1:]:
        x, y = grid.grid_to_world(row, col)
        world_path.append((float(x), float(y)))

    return world_path


def path_is_blocked(grid, costmap, pose, path):
    """Check the complete cached path against the latest global costmap."""
    if path is None:
        return True

    if not path:
        return False

    current = grid.world_to_grid(pose.x, pose.y)
    if current is None:
        return True

    current = tuple(current)
    if not _in_bounds(costmap.shape, current):
        return True

    checking_costmap = costmap.copy()
    checking_costmap[current] = 0
    previous = current

    for waypoint in path:
        waypoint_cell = grid.world_to_grid(waypoint[0], waypoint[1])
        if waypoint_cell is None:
            return True

        waypoint_cell = tuple(waypoint_cell)
        if not _in_bounds(checking_costmap.shape, waypoint_cell):
            return True

        if not _line_of_sight(checking_costmap, previous, waypoint_cell):
            return True

        previous = waypoint_cell

    return False
