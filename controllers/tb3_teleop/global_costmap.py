"""Global costmap conversion for TECH WEEK AMR.

Input occupancy convention expected from the SLAM/mapping module:
    -1   unknown
     0   free
   100   occupied

Output costmap convention follows the lecture material:
     0       free
     1..252  traversable inflation cost
     253     inscribed/collision area
     254     occupied obstacle
     255     unknown
"""

import heapq
import math

import numpy as np


FREE_COST = np.uint8(0)
INSCRIBED_COST = np.uint8(253)
OCCUPIED_COST = np.uint8(254)
UNKNOWN_COST = np.uint8(255)
MAX_INFLATION_COST = 252


_NEIGHBORS_8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def build_global_costmap(
    occupancy,
    resolution,
    robot_radius=0.105,
    safety_clearance=0.12,
):
    """Convert an occupancy grid into an inflated global costmap.

    ``robot_radius`` creates a hard collision region around obstacles.
    ``safety_clearance`` creates a traversable 1..252 inflation band outside it.
    Unknown cells stay blocked as 255.
    """

    occupancy = np.asarray(occupancy)

    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a 2-D array")

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    rows, cols = occupancy.shape

    costmap = np.zeros((rows, cols), dtype=np.uint8)
    costmap[occupancy < 0] = UNKNOWN_COST
    costmap[occupancy >= 100] = OCCUPIED_COST

    occupied_cells = np.argwhere(occupancy >= 100)
    if len(occupied_cells) == 0:
        return costmap

    total_radius = max(robot_radius + safety_clearance, robot_radius)

    # Distance-to-nearest-obstacle only needs to be computed inside inflation radius.
    distance = np.full((rows, cols), np.inf, dtype=float)
    heap = []

    for row, col in occupied_cells:
        row = int(row)
        col = int(col)
        distance[row, col] = 0.0
        heapq.heappush(heap, (0.0, row, col))

    while heap:
        dist, row, col = heapq.heappop(heap)

        if dist != distance[row, col]:
            continue

        if dist > total_radius:
            continue

        for dr, dc, step_cells in _NEIGHBORS_8:
            nr = row + dr
            nc = col + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            next_dist = dist + step_cells * resolution

            if next_dist > total_radius:
                continue

            if next_dist < distance[nr, nc]:
                distance[nr, nc] = next_dist
                heapq.heappush(heap, (next_dist, nr, nc))

    free_mask = occupancy == 0

    # Center of the robot may not enter this band without geometric collision.
    inscribed_mask = (
        free_mask
        & (distance > 0.0)
        & (distance <= robot_radius)
    )
    costmap[inscribed_mask] = INSCRIBED_COST

    # Outside the robot body radius, use a graded traversable inflation band.
    if safety_clearance > 0.0:
        inflation_mask = (
            free_mask
            & (distance > robot_radius)
            & (distance <= total_radius)
        )

        inflation_dist = distance[inflation_mask]
        ratio = (total_radius - inflation_dist) / safety_clearance
        values = 1 + np.rint(ratio * (MAX_INFLATION_COST - 1))
        values = np.clip(values, 1, MAX_INFLATION_COST).astype(np.uint8)
        costmap[inflation_mask] = values

    return costmap
