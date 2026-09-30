"""Rolling-window local costmap built from the TurtleBot3 LDS-01 range scan."""

import math

import numpy as np

from global_costmap import build_global_costmap, INSCRIBED_COST


class LocalCostmap:
    """Robot-centered costmap in the robot frame (+x forward, +y left)."""

    def __init__(
        self,
        window_size_m=2.4,
        resolution=0.05,
        robot_radius=0.105,
        safety_clearance=0.10,
    ):
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.safety_clearance = float(safety_clearance)

        size = int(math.ceil(window_size_m / self.resolution))
        if size % 2 == 0:
            size += 1

        self.rows = size
        self.cols = size
        self.center_row = size // 2
        self.center_col = size // 2
        self.costmap = np.zeros((size, size), dtype=np.uint8)

    def robot_to_grid(self, x, y):
        col = self.center_col + int(round(x / self.resolution))
        row = self.center_row - int(round(y / self.resolution))

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None

        return row, col

    def update(self, ranges):
        """Rebuild the rolling local costmap from a 360-degree range image.

        The provided starter repo uses:
            index 180 -> front
            index  90 -> left
            index 270 -> right
            index   0 -> back
        """
        occupancy = np.zeros((self.rows, self.cols), dtype=np.int8)

        if ranges is None:
            self.costmap.fill(0)
            return self.costmap

        count = len(ranges)
        if count == 0:
            self.costmap.fill(0)
            return self.costmap

        max_endpoint_range = (
            min(self.rows, self.cols) // 2 - 1
        ) * self.resolution

        for index, measured_range in enumerate(ranges):
            if not math.isfinite(measured_range) or measured_range <= 0.0:
                continue

            # If the hit is outside our rolling window, it cannot influence the
            # local controller inside this window.
            if measured_range >= max_endpoint_range:
                continue

            angle = math.pi - (2.0 * math.pi * index / count)
            x = measured_range * math.cos(angle)
            y = measured_range * math.sin(angle)
            cell = self.robot_to_grid(x, y)

            if cell is not None:
                occupancy[cell] = 100

        self.costmap = build_global_costmap(
            occupancy,
            self.resolution,
            robot_radius=self.robot_radius,
            safety_clearance=self.safety_clearance,
        )
        return self.costmap

    def cost_at(self, x, y):
        cell = self.robot_to_grid(x, y)
        if cell is None:
            return 255
        return int(self.costmap[cell])

    def trajectory_score(self, linear, angular, horizon=1.0, dt=0.10):
        """Return (safe, normalized_obstacle_cost) for a candidate trajectory.

        The robot can occasionally already be standing in a center grid cell
        marked INSCRIBED because obstacle inflation is discretized onto a
        5-cm grid.  Rejecting that cell immediately would also reject every
        in-place recovery rotation.  Therefore only the *current center cell*
        gets an INSCRIBED exception while the robot is escaping it.

        OCCUPIED (254), UNKNOWN (255), and every other INSCRIBED cell remain
        blocked.
        """
        x = 0.0
        y = 0.0
        theta = 0.0
        accumulated = 0.0
        samples = 0

        steps = max(1, int(math.ceil(horizon / dt)))
        start_cell = self.robot_to_grid(0.0, 0.0)

        for _ in range(steps):
            if abs(angular) < 1e-9:
                x += linear * math.cos(theta) * dt
                y += linear * math.sin(theta) * dt
            else:
                theta_mid = theta + 0.5 * angular * dt
                x += linear * math.cos(theta_mid) * dt
                y += linear * math.sin(theta_mid) * dt
                theta += angular * dt

            cell = self.robot_to_grid(x, y)
            if cell is None:
                return False, math.inf

            cost = int(self.costmap[cell])

            # Start-cell exception: the physical robot is already here.
            # Allow only an INSCRIBED (253) center cell so the controller can
            # rotate or move out.  A real occupied/unknown cell is never freed.
            if (
                cell == start_cell
                and cost == int(INSCRIBED_COST)
            ):
                cost = 0

            if cost >= int(INSCRIBED_COST):
                return False, math.inf

            accumulated += cost / 252.0
            samples += 1

        return True, accumulated / max(samples, 1)

