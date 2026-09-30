from collections import deque
import math

import numpy as np
from PIL import Image


UNKNOWN = -1
FREE = 0
OCCUPIED = 100

FREE_UPDATE = -0.35
OCCUPIED_UPDATE = 0.85
MIN_LOG_ODDS = -20.0
MAX_LOG_ODDS = 20.0
FREE_THRESHOLD = -3.6
OCCUPIED_THRESHOLD = 6.0

LIDAR_X = 0.0
LIDAR_Y = 0.0
LIDAR_Z = 0.25
MIN_OBSTACLE_HEIGHT = 0.12
MAX_OBSTACLE_HEIGHT = 1.40
MIN_RANGE = 0.05
MAX_RANGE = 8.0
NO_RETURN_MARGIN = 0.15

LIDAR_BEARING_OFFSET = 0.0

FREE_SAMPLE_FACTOR = 0.5

SAFETY_RADIUS = 0.5

SMEAR_DEVIATION = 0.10
SMEAR_MIN_WEIGHT = 0.05

MINIMUM_TRAVEL_DISTANCE = 0.25
MINIMUM_TRAVEL_HEADING = 0.20
SCAN_BUFFER_STEPS = 4
MIN_OCCUPIED_FOR_MATCH = 120

SCAN_MEDIAN_BIN_RAD = math.radians(1.0)
SCAN_MEDIAN_GATE_M = 0.30
SCAN_MEDIAN_MIN_POINTS = 3


class OccupancyGrid:
    def __init__(self, rows=121, cols=121, resolution=0.1,
                 localization=None, scan_matching=True):
        if rows <= 0 or cols <= 0:
            raise ValueError("격자 크기는 1 이상.")
        if resolution <= 0:
            raise ValueError("해상도는 0보다 커야함.")

        self.rows = int(rows)
        self.cols = int(cols)
        self.resolution = float(resolution)
        self.origin_row = self.rows // 2
        self.origin_col = self.cols // 2
        self.data = np.full((self.rows, self.cols), UNKNOWN, dtype=np.int8)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.last_point_time = -math.inf
        self.revision = 0

        self.localization = localization
        self.matcher = CorrelativeScanMatcher(self) if scan_matching else None
        self._buffer = deque(maxlen=SCAN_BUFFER_STEPS)
        self._last_match_pose = None
        self.match_count = 0
        self.failed_match_count = 0
        self.last_response = 0.0
        self.last_correction = (0.0, 0.0, 0.0)

        self._field = None
        self._field_revision = -1
        self._safe_grid = None
        self._safe_key = None

    def bind_localization(self, localization):
        self.localization = localization

    def is_inside(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def world_to_grid(self, x, y):
        col = self.origin_col + math.floor(float(x) / self.resolution + 0.5)
        row = self.origin_row + math.floor(float(y) / self.resolution + 0.5)
        return (row, col) if self.is_inside(row, col) else None

    def grid_to_world(self, row, col):
        if not self.is_inside(row, col):
            return None

        x = (col - self.origin_col) * self.resolution
        y = (row - self.origin_row) * self.resolution
        return x, y

    def update(self, pose, lidar_points):
        x, y, theta = float(pose.x), float(pose.y), float(pose.theta)
        if not all(math.isfinite(value) for value in (x, y, theta)):
            raise ValueError("로봇 자세에 유효하지 않은 값이 있습니다.")

        points, is_obstacle = self._collect_points(lidar_points)

        if points.size and is_obstacle.any():
            self._buffer.append(
                _to_world(points[is_obstacle], x, y, theta))

        corrected = self._try_scan_match(pose)
        if corrected is not None:
            x, y, theta = corrected

        self._integrate(points, is_obstacle, x, y, theta)
        self._refresh_data()
        return self.data

    def make_safe_grid(self, clearance_m: float) -> np.ndarray:
        radius = SAFETY_RADIUS + max(0.0, float(clearance_m))
        key = (self.revision, round(radius, 4))
        if self._safe_key == key:
            return self._safe_grid

        lethal = (self.data == OCCUPIED)
        cells = int(math.floor(radius / self.resolution + 0.5))
        blocked = _inflate(lethal, cells)
        blocked |= (self.data == UNKNOWN)

        self._safe_grid = blocked
        self._safe_key = key
        return blocked

    def likelihood_field(self):
        if self._field_revision == self.revision:
            return self._field

        occupied = (self.log_odds >= OCCUPIED_THRESHOLD)
        self._field_revision = self.revision
        if int(occupied.sum()) < MIN_OCCUPIED_FOR_MATCH:
            self._field = None
            return None

        source = occupied.astype(np.float64)
        field = source.copy()
        radius = max(1, int(round(2.0 * SMEAR_DEVIATION / self.resolution)))
        for row_shift in range(-radius, radius + 1):
            for col_shift in range(-radius, radius + 1):
                if row_shift == 0 and col_shift == 0:
                    continue
                distance = math.hypot(row_shift, col_shift) * self.resolution
                weight = math.exp(-0.5 * (distance / SMEAR_DEVIATION) ** 2)
                if weight < SMEAR_MIN_WEIGHT:
                    continue
                _max_shift(field, source, row_shift, col_shift, weight)

        self._field = field
        return field

    def save_png(self, path):
        pixels = np.full(self.data.shape, 128, dtype=np.uint8)
        pixels[self.data == FREE] = 255
        pixels[self.data == OCCUPIED] = 0
        Image.fromarray(np.flipud(pixels).copy()).save(path)

    def save_safe_png(self, path, clearance_m=0.0):
        safe = self.make_safe_grid(clearance_m)
        pixels = np.full(self.data.shape, 255, dtype=np.uint8)
        pixels[safe] = 90
        pixels[self.data == UNKNOWN] = 128
        pixels[self.data == OCCUPIED] = 0
        Image.fromarray(np.flipud(pixels).copy()).save(path)

    def stats(self):
        free = int((self.data == FREE).sum())
        occupied = int((self.data == OCCUPIED).sum())
        return (f'free={free} occ={occupied} '
                f'match={self.match_count}/{self.match_count + self.failed_match_count} '
                f'response={self.last_response:.2f}')

    def _collect_points(self, lidar_points):
        xs = []
        ys = []
        flags = []
        newest = self.last_point_time
        offset_cos = math.cos(LIDAR_BEARING_OFFSET)
        offset_sin = math.sin(LIDAR_BEARING_OFFSET)

        for point in lidar_points:
            try:
                local_x = float(point.x)
                local_y = float(point.y)
                local_z = float(point.z)
            except (AttributeError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in
                       (local_x, local_y, local_z)):
                continue

            point_time = getattr(point, 'time', None)
            if point_time is not None:
                try:
                    point_time = float(point_time)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(point_time):
                    continue
                if point_time <= self.last_point_time + 1e-6:
                    continue
                if point_time > newest:
                    newest = point_time

            local_x, local_y = (offset_cos * local_x - offset_sin * local_y,
                                offset_sin * local_x + offset_cos * local_y)

            planar_range = math.hypot(local_x, local_y)
            if planar_range < MIN_RANGE:
                continue

            height = LIDAR_Z + local_z
            full_range = math.hypot(planar_range, local_z)
            is_obstacle = (MIN_OBSTACLE_HEIGHT <= height <= MAX_OBSTACLE_HEIGHT
                           and full_range < MAX_RANGE - NO_RETURN_MARGIN)

            xs.append(local_x)
            ys.append(local_y)
            flags.append(is_obstacle)

        self.last_point_time = newest
        if not xs:
            return np.empty((0, 2)), np.empty(0, dtype=bool)
        return (np.column_stack((np.asarray(xs), np.asarray(ys))),
                np.asarray(flags, dtype=bool))

    def _try_scan_match(self, pose):
        if self.matcher is None:
            return None

        if self._last_match_pose is None:
            self._last_match_pose = (pose.x, pose.y, pose.theta)
            return None

        last_x, last_y, last_theta = self._last_match_pose
        moved = math.hypot(pose.x - last_x, pose.y - last_y)
        turned = abs(_wrap(pose.theta - last_theta))
        if moved < MINIMUM_TRAVEL_DISTANCE and turned < MINIMUM_TRAVEL_HEADING:
            return None
        if not self._buffer:
            return None

        scan = _to_robot(np.concatenate(self._buffer),
                         pose.x, pose.y, pose.theta)
        if SCAN_MEDIAN_BIN_RAD > 0.0:
            scan = _median_by_bearing(scan, SCAN_MEDIAN_BIN_RAD)
        result = self.matcher.match(pose.x, pose.y, pose.theta, scan)
        self._buffer.clear()

        if result is None:
            self.failed_match_count += 1
            self.last_response = self.matcher.last_response
            self._last_match_pose = (pose.x, pose.y, pose.theta)
            return None

        matched_x, matched_y, matched_theta, response = result
        self.match_count += 1
        self.last_response = response
        self.last_correction = (matched_x - pose.x, matched_y - pose.y,
                                _wrap(matched_theta - pose.theta))

        if self.localization is not None:
            fused = self.localization.update_scan_match(
                matched_x, matched_y, matched_theta, response)
            pose.x, pose.y, pose.theta = fused.x, fused.y, fused.theta
        else:
            pose.x, pose.y, pose.theta = matched_x, matched_y, matched_theta

        self._last_match_pose = (pose.x, pose.y, pose.theta)
        return pose.x, pose.y, pose.theta

    def _integrate(self, points, is_obstacle, x, y, theta):
        if points.size == 0:
            return

        cosine = math.cos(theta)
        sine = math.sin(theta)
        sensor_x = x + cosine * LIDAR_X - sine * LIDAR_Y
        sensor_y = y + sine * LIDAR_X + cosine * LIDAR_Y

        hit_x = sensor_x + cosine * points[:, 0] - sine * points[:, 1]
        hit_y = sensor_y + sine * points[:, 0] + cosine * points[:, 1]
        distance = np.hypot(hit_x - sensor_x, hit_y - sensor_y)

        reach = np.where(is_obstacle,
                         np.maximum(0.0, distance - 0.5 * self.resolution),
                         distance)
        step = self.resolution * FREE_SAMPLE_FACTOR
        counts = np.maximum(1, np.ceil(reach / step).astype(np.int64))
        longest = int(counts.max())

        ticks = np.arange(longest)
        fraction = ticks[None, :] / counts[:, None]
        alive = ticks[None, :] < counts[:, None]
        scale = np.divide(reach, distance, out=np.zeros_like(reach),
                          where=distance > 0.0)
        sample_x = sensor_x + fraction * (scale * (hit_x - sensor_x))[:, None]
        sample_y = sensor_y + fraction * (scale * (hit_y - sensor_y))[:, None]

        free_rows, free_cols = self._to_cells(sample_x, sample_y)
        inside = alive & self._inside(free_rows, free_cols)
        if inside.any():
            flat = np.unique(free_rows[inside] * self.cols + free_cols[inside])
            rows, cols = flat // self.cols, flat % self.cols
            self.log_odds[rows, cols] = np.maximum(
                MIN_LOG_ODDS, self.log_odds[rows, cols] + FREE_UPDATE)

        hit_rows, hit_cols = self._to_cells(hit_x, hit_y)
        marked = is_obstacle & self._inside(hit_rows, hit_cols)
        if marked.any():
            flat = np.unique(hit_rows[marked] * self.cols + hit_cols[marked])
            rows, cols = flat // self.cols, flat % self.cols
            self.log_odds[rows, cols] = np.minimum(
                MAX_LOG_ODDS, self.log_odds[rows, cols] + OCCUPIED_UPDATE)

    def _to_cells(self, x, y):
        col = self.origin_col + np.floor(
            x / self.resolution + 0.5).astype(np.int64)
        row = self.origin_row + np.floor(
            y / self.resolution + 0.5).astype(np.int64)
        return row, col

    def _inside(self, row, col):
        return ((row >= 0) & (row < self.rows) &
                (col >= 0) & (col < self.cols))

    def _refresh_data(self):
        updated = np.full(self.data.shape, UNKNOWN, dtype=np.int8)
        updated[self.log_odds <= FREE_THRESHOLD] = FREE
        updated[self.log_odds >= OCCUPIED_THRESHOLD] = OCCUPIED
        if not np.array_equal(updated, self.data):
            self.data = updated
            self.revision += 1


_BAND_CACHE = {}


def _disk_bands(radius_cells):
    if radius_cells in _BAND_CACHE:
        return _BAND_CACHE[radius_cells]

    bands = {}
    for row_shift in range(-radius_cells, radius_cells + 1):
        span = radius_cells * radius_cells - row_shift * row_shift
        width = int(math.floor(math.sqrt(span))) if span > 0 else 0
        bands.setdefault(width, []).append(row_shift)
    _BAND_CACHE[radius_cells] = bands
    return bands


def _dilate_rows(mask, width):
    if width <= 0:
        return mask
    cols = mask.shape[1]
    counts = np.zeros((mask.shape[0], cols + 1), dtype=np.int32)
    np.cumsum(mask, axis=1, out=counts[:, 1:])
    columns = np.arange(cols)
    low = np.clip(columns - width, 0, cols)
    high = np.clip(columns + width + 1, 0, cols)
    return (counts[:, high] - counts[:, low]) > 0


def _inflate(mask, radius_cells):
    if radius_cells <= 0:
        return mask.copy()

    blocked = np.zeros_like(mask)
    for width, row_shifts in _disk_bands(radius_cells).items():
        band = _dilate_rows(mask, width)
        for row_shift in row_shifts:
            _or_shift(blocked, band, row_shift, 0)
    return blocked


def _shift_slices(shape, row_shift, col_shift):
    rows, cols = shape
    row_start, row_end = max(row_shift, 0), rows + min(row_shift, 0)
    col_start, col_end = max(col_shift, 0), cols + min(col_shift, 0)
    if row_start >= row_end or col_start >= col_end:
        return None
    destination = (slice(row_start, row_end), slice(col_start, col_end))
    source = (slice(row_start - row_shift, row_end - row_shift),
              slice(col_start - col_shift, col_end - col_shift))
    return destination, source


def _or_shift(destination, source, row_shift, col_shift):
    slices = _shift_slices(source.shape, row_shift, col_shift)
    if slices is None:
        return
    target, origin = slices
    destination[target] |= source[origin]


def _max_shift(destination, source, row_shift, col_shift, weight):
    slices = _shift_slices(source.shape, row_shift, col_shift)
    if slices is None:
        return
    target, origin = slices
    np.maximum(destination[target], source[origin] * weight,
               out=destination[target])


def _to_world(points, x, y, theta):
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.column_stack((
        x + cosine * points[:, 0] - sine * points[:, 1],
        y + sine * points[:, 0] + cosine * points[:, 1]))


def _to_robot(points, x, y, theta):
    cosine, sine = math.cos(theta), math.sin(theta)
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    return np.column_stack((cosine * dx + sine * dy,
                            -sine * dx + cosine * dy))


def _median_by_bearing(points, bin_rad):
    if len(points) == 0:
        return points
    bearing = np.arctan2(points[:, 1], points[:, 0])
    distance = np.hypot(points[:, 0], points[:, 1])
    bins = np.floor(bearing / bin_rad).astype(np.int64)
    order = np.lexsort((distance, bins))
    bins, bearing, distance = bins[order], bearing[order], distance[order]
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    ends = np.r_[starts[1:], len(bins)]

    out = []
    for start, end in zip(starts, ends):
        if end - start < SCAN_MEDIAN_MIN_POINTS:
            continue
        near = distance[start:end]
        keep = np.abs(near - near[(end - start) // 2]) <= SCAN_MEDIAN_GATE_M
        angle = float(np.mean(bearing[start:end][keep]))
        reach = float(np.mean(near[keep]))
        out.append((reach * math.cos(angle), reach * math.sin(angle)))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


COARSE_RANGE_M = 0.30
COARSE_STEP_M = 0.06
COARSE_ANGLE_RAD = 0.30
COARSE_ANGLE_STEP_RAD = 0.05

FINE_RANGE_M = 0.06
FINE_STEP_M = 0.015
FINE_ANGLE_RAD = 0.05
FINE_ANGLE_STEP_RAD = 0.0125

MINIMUM_RESPONSE = 0.35

MAX_CORRECTION_M = 0.15
MAX_CORRECTION_RAD = 0.10

REFINE_ITERATIONS = 5
REFINE_DAMPING = 0.1
REFINE_MAX_STEP_M = 0.05
REFINE_MAX_STEP_RAD = 0.03

MAX_POINTS = 250
MIN_POINTS = 60


class CorrelativeScanMatcher:
    def __init__(self, grid):
        self.grid = grid
        self.last_response = 0.0
        self.last_correction = (0.0, 0.0, 0.0)
        self.rejected_corrections = 0

    def match(self, x, y, theta, points):
        points = _thin(points)
        if points is None:
            return None

        field = self.grid.likelihood_field()
        if field is None:
            return None

        coarse = self._search(field, points, x, y, theta,
                              COARSE_RANGE_M, COARSE_STEP_M,
                              COARSE_ANGLE_RAD, COARSE_ANGLE_STEP_RAD,
                              refine=False)
        fine = self._search(field, points, coarse[0], coarse[1], coarse[2],
                            FINE_RANGE_M, FINE_STEP_M,
                            FINE_ANGLE_RAD, FINE_ANGLE_STEP_RAD,
                            refine=True)

        response = fine[3] / len(points)
        self.last_response = response
        if response < MINIMUM_RESPONSE:
            return None

        if REFINE_ITERATIONS > 0:
            fine = self._refine(field, points, *fine[:3]) + (fine[3],)

        correction = (fine[0] - x, fine[1] - y, _wrap(fine[2] - theta))
        if (math.hypot(correction[0], correction[1]) > MAX_CORRECTION_M or
                abs(correction[2]) > MAX_CORRECTION_RAD):
            self.rejected_corrections += 1
            return None

        self.last_correction = correction
        return fine[0], fine[1], fine[2], response

    def _search(self, field, points, x, y, theta,
                range_m, step_m, range_rad, step_rad, refine):
        grid = self.grid
        resolution = grid.resolution
        rows, cols = grid.rows, grid.cols

        offsets = np.arange(-range_m, range_m + 1e-9, step_m)
        deltas = np.arange(-range_rad, range_rad + 1e-9, step_rad)
        span = len(offsets)
        shift_x, shift_y = np.meshgrid(offsets, offsets, indexing='ij')
        base_x = (x + shift_x).ravel()
        base_y = (y + shift_y).ravel()

        point_x = points[:, 0]
        point_y = points[:, 1]
        cube = np.empty((len(deltas), span, span))

        for index, delta in enumerate(deltas):
            cosine = math.cos(theta + delta)
            sine = math.sin(theta + delta)
            rotated_x = cosine * point_x - sine * point_y
            rotated_y = sine * point_x + cosine * point_y

            world_x = base_x[:, None] + rotated_x[None, :]
            world_y = base_y[:, None] + rotated_y[None, :]
            col = grid.origin_col + np.floor(
                world_x / resolution + 0.5).astype(np.int32)
            row = grid.origin_row + np.floor(
                world_y / resolution + 0.5).astype(np.int32)
            inside = ((row >= 0) & (row < rows) & (col >= 0) & (col < cols))
            np.clip(row, 0, rows - 1, out=row)
            np.clip(col, 0, cols - 1, out=col)
            cube[index] = np.where(inside, field[row, col], 0.0).sum(
                axis=1).reshape(span, span)

        angle_index, x_index, y_index = np.unravel_index(
            int(np.argmax(cube)), cube.shape)
        peak = float(cube[angle_index, x_index, y_index])
        best_x = float(offsets[x_index])
        best_y = float(offsets[y_index])
        best_angle = float(deltas[angle_index])

        if refine:
            best_x += step_m * _vertex(
                _at(cube, angle_index, x_index - 1, y_index), peak,
                _at(cube, angle_index, x_index + 1, y_index))
            best_y += step_m * _vertex(
                _at(cube, angle_index, x_index, y_index - 1), peak,
                _at(cube, angle_index, x_index, y_index + 1))
            best_angle += step_rad * _vertex(
                _at(cube, angle_index - 1, x_index, y_index), peak,
                _at(cube, angle_index + 1, x_index, y_index))

        return (x + best_x, y + best_y, _wrap(theta + best_angle), peak)

    def _refine(self, field, points, x, y, theta):
        grid = self.grid
        resolution = grid.resolution
        start = (x, y, theta)
        point_x = points[:, 0]
        point_y = points[:, 1]

        for _ in range(REFINE_ITERATIONS):
            cosine, sine = math.cos(theta), math.sin(theta)
            world_x = x + cosine * point_x - sine * point_y
            world_y = y + sine * point_x + cosine * point_y
            cell_x = world_x / resolution + grid.origin_col
            cell_y = world_y / resolution + grid.origin_row
            col = np.floor(cell_x).astype(np.int64)
            row = np.floor(cell_y).astype(np.int64)
            inside = ((row >= 0) & (row < grid.rows - 1) &
                      (col >= 0) & (col < grid.cols - 1))
            if int(inside.sum()) < MIN_POINTS:
                break
            col, row = col[inside], row[inside]
            fx, fy = cell_x[inside] - col, cell_y[inside] - row
            m00, m10 = field[row, col], field[row, col + 1]
            m01, m11 = field[row + 1, col], field[row + 1, col + 1]

            value = ((1 - fy) * ((1 - fx) * m00 + fx * m10) +
                     fy * ((1 - fx) * m01 + fx * m11))
            grad_x = ((1 - fy) * (m10 - m00) + fy * (m11 - m01)) / resolution
            grad_y = ((1 - fx) * (m01 - m00) + fx * (m11 - m10)) / resolution
            px, py = point_x[inside], point_y[inside]
            grad_theta = (grad_x * (-sine * px - cosine * py) +
                          grad_y * (cosine * px - sine * py))

            jacobian = np.column_stack((grad_x, grad_y, grad_theta))
            hessian = jacobian.T @ jacobian
            hessian += REFINE_DAMPING * np.diag(np.diag(hessian) + 1e-9)
            try:
                step = np.linalg.solve(hessian, jacobian.T @ (1.0 - value))
            except np.linalg.LinAlgError:
                break
            x, y, theta = x + step[0], y + step[1], _wrap(theta + step[2])
            if abs(step[0]) + abs(step[1]) < 1e-4 and abs(step[2]) < 1e-4:
                break

        if (math.hypot(x - start[0], y - start[1]) > REFINE_MAX_STEP_M or
                abs(_wrap(theta - start[2])) > REFINE_MAX_STEP_RAD):
            return start
        return x, y, theta


def _at(cube, angle_index, x_index, y_index):
    if not (0 <= angle_index < cube.shape[0] and
            0 <= x_index < cube.shape[1] and
            0 <= y_index < cube.shape[2]):
        return None
    return float(cube[angle_index, x_index, y_index])


def _vertex(low, peak, high):
    if low is None or high is None:
        return 0.0
    curvature = low - 2.0 * peak + high
    if curvature >= -1e-12:
        return 0.0
    offset = 0.5 * (low - high) / curvature
    return max(-0.5, min(0.5, offset))


def _thin(points):
    if points is None:
        return None
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    if len(points) < MIN_POINTS:
        return None
    if len(points) > MAX_POINTS:
        step = len(points) / MAX_POINTS
        index = (np.arange(MAX_POINTS) * step).astype(np.int64)
        points = points[index]
    return points
