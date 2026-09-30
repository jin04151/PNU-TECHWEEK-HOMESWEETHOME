from dataclasses import dataclass
import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160

X, Y, THETA, V, OMEGA, GYRO_BIAS = range(6)
STATE_SIZE = 6

COMMAND_TRACKING = 0.0

PROCESS_STD = np.array([0.01, 0.01, 0.01, 0.35, 1.2, 0.0])
GYRO_BIAS_WALK_STD = 0.00001

ENCODER_V_STD = 0.02
ENCODER_OMEGA_STD = 5.0
GYRO_OMEGA_STD = 0.002
COMPASS_THETA_STD = 0.05
COMPASS_ADAPT_ALPHA = 0.01
COMPASS_MIN_THETA_STD = 0.005
SCAN_MATCH_XY_STD = 0.03
SCAN_MATCH_THETA_STD = 0.02

ACCEL_LATERAL_STD = 0.08
ACCEL_BIAS_ALPHA = 0.02
ACCEL_MIN_SPEED = 0.05
ACCEL_GATE = 0.6

ZUPT_V_STD = 0.002
ZUPT_OMEGA_STD = 0.002

INITIAL_STD = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

STATIONARY_DISTANCE = 1e-4

COMPASS_MIN_NORM = 1e-3
CALIBRATION_TURN_RAD = 0.6
CALIBRATION_MAX_SAMPLES = 400

MAX_DT = 1.0


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class Localization:
    def __init__(self, initial_pose=None):
        start = initial_pose if initial_pose is not None else Pose()
        self.state = np.array(
            [start.x, start.y, wrap_angle(start.theta), 0.0, 0.0, 0.0],
            dtype=float)
        self.covariance = np.diag(INITIAL_STD ** 2)
        self.pose = Pose(start.x, start.y, wrap_angle(start.theta))

        self.previous_left = None
        self.previous_right = None
        self.last_update_time = None
        self.travelled = 0.0

        self.command = None

        self.gyro = (0.0, 0.0, 0.0)
        self.acceleration = (0.0, 0.0, 0.0)
        self.compass = (0.0, 0.0, 0.0)

        self.stationary = False

        self.compass_sign = None
        self.compass_offset = 0.0
        self.compass_heading = None
        self.compass_variance = COMPASS_THETA_STD ** 2
        self._calibration_samples = []
        self._calibration_previous_raw = None
        self._calibration_turn = 0.0
        self._calibration_correlation = 0.0

        self.scan_match_count = 0
        self.landmark_count = 0

        self.accel_bias = None
        self.accel_count = 0
        self.camera = CameraLandmarks()

    def set_command(self, linear, angular):
        if math.isfinite(linear) and math.isfinite(angular):
            self.command = (float(linear), float(angular))

    def update(self, left_angle, right_angle, gyro, acceleration, compass, now,
               command=None):
        self.gyro = _as_triple(gyro)
        self.acceleration = _as_triple(acceleration)
        self.compass = _as_triple(compass)
        if command is not None:
            self.set_command(command[0], command[1])

        left_angle = float(left_angle)
        right_angle = float(right_angle)

        if self.previous_left is None:
            self.previous_left = left_angle
            self.previous_right = right_angle
            self.last_update_time = float(now)
            return self.pose

        dt = float(now) - self.last_update_time
        self.last_update_time = float(now)
        if not (0.0 < dt <= MAX_DT):
            self.previous_left = left_angle
            self.previous_right = right_angle
            return self.pose

        left_distance = (left_angle - self.previous_left) * WHEEL_RADIUS
        right_distance = (right_angle - self.previous_right) * WHEEL_RADIUS
        self.previous_left = left_angle
        self.previous_right = right_angle
        if not (math.isfinite(left_distance) and math.isfinite(right_distance)):
            return self._sync_pose()

        self._predict_velocity(dt)

        encoder_v = 0.5 * (left_distance + right_distance) / dt
        encoder_omega = (right_distance - left_distance) / (WHEEL_BASE * dt)
        self._update_linear(
            np.array([encoder_v, encoder_omega]),
            _rows(V, OMEGA),
            np.diag([ENCODER_V_STD ** 2, ENCODER_OMEGA_STD ** 2]))

        moving = (abs(left_distance) > STATIONARY_DISTANCE or
                  abs(right_distance) > STATIONARY_DISTANCE)
        self._update_gyro(moving)

        self._update_accelerometer(moving)

        self._integrate_pose(dt)

        self._update_compass(dt)

        self.travelled += abs(0.5 * (left_distance + right_distance))
        return self._sync_pose()

    def update_scan_match(self, x, y, theta, response=1.0):
        response = min(1.0, max(1e-3, float(response)))
        xy_variance = (SCAN_MATCH_XY_STD / response) ** 2
        theta_variance = (SCAN_MATCH_THETA_STD / response) ** 2
        self._update_linear(
            np.array([float(x), float(y), wrap_angle(float(theta))]),
            _rows(X, Y, THETA),
            np.diag([xy_variance, xy_variance, theta_variance]),
            angle_rows=(2,))
        self.scan_match_count += 1
        return self._sync_pose()

    def update_landmark(self, landmark_xy, observed_range, observed_bearing,
                        range_std=0.05, bearing_std=0.02):
        lx, ly = float(landmark_xy[0]), float(landmark_xy[1])
        dx = lx - self.state[X]
        dy = ly - self.state[Y]
        squared = dx * dx + dy * dy
        distance = math.sqrt(squared)
        if distance < 1e-3:
            return self.pose

        predicted = np.array(
            [distance, wrap_angle(math.atan2(dy, dx) - self.state[THETA])])

        jacobian = np.zeros((2, STATE_SIZE))
        jacobian[0, X] = -dx / distance
        jacobian[0, Y] = -dy / distance
        jacobian[1, X] = dy / squared
        jacobian[1, Y] = -dx / squared
        jacobian[1, THETA] = -1.0

        measurement = np.array(
            [float(observed_range), wrap_angle(float(observed_bearing))])
        noise = np.diag([float(range_std) ** 2, float(bearing_std) ** 2])
        self._apply_update(measurement - predicted, jacobian, noise,
                           angle_rows=(1,))
        self.landmark_count += 1
        return self._sync_pose()

    def update_camera(self, image, width, height, fov, lidar_points,
                      bearing_offset=0.0):
        detections = self.camera.detect(image, int(width), int(height), float(fov),
                                        lidar_points, bearing_offset)
        if not detections:
            return self.pose
        x, y, theta = self.state[X], self.state[Y], self.state[THETA]
        cosine, sine = math.cos(theta), math.sin(theta)
        confident = max(self.position_std()[:2]) < LANDMARK_REGISTER_MAX_STD

        for color, local_x, local_y, _ in detections:
            world_x = x + cosine * local_x - sine * local_y
            world_y = y + sine * local_x + cosine * local_y
            landmark = self.camera.associate(color, world_x, world_y)
            if landmark is None:
                if confident:
                    self.camera.landmarks.append(dict(
                        color=color, sum_x=world_x, sum_y=world_y, count=1,
                        x=world_x, y=world_y, fixed=False))
                continue
            if not landmark['fixed']:
                if confident:
                    landmark['sum_x'] += world_x
                    landmark['sum_y'] += world_y
                    landmark['count'] += 1
                    landmark['x'] = landmark['sum_x'] / landmark['count']
                    landmark['y'] = landmark['sum_y'] / landmark['count']
                    landmark['fixed'] = landmark['count'] >= LANDMARK_REGISTER_COUNT
                continue

            observed_range = math.hypot(local_x, local_y)
            observed_bearing = math.atan2(local_y, local_x)
            dx, dy = landmark['x'] - self.state[X], landmark['y'] - self.state[Y]
            predicted_range = math.hypot(dx, dy)
            predicted_bearing = wrap_angle(math.atan2(dy, dx) - self.state[THETA])
            if (abs(observed_range - predicted_range) > LANDMARK_GATE[0] or
                    abs(wrap_angle(observed_bearing - predicted_bearing)) > LANDMARK_GATE[1]):
                self.camera.rejected += 1
                continue
            self.update_landmark((landmark['x'], landmark['y']), observed_range,
                                 observed_bearing, LANDMARK_RANGE_STD,
                                 LANDMARK_BEARING_STD)
            self.camera.used += 1
        return self._sync_pose()

    def update_position(self, x, y, std=0.1):
        self._update_linear(
            np.array([float(x), float(y)]),
            _rows(X, Y),
            np.diag([float(std) ** 2, float(std) ** 2]))
        return self._sync_pose()

    @property
    def gyro_bias(self):
        return float(self.state[GYRO_BIAS])

    def position_std(self):
        diagonal = np.clip(np.diag(self.covariance), 0.0, None)
        return (math.sqrt(diagonal[X]), math.sqrt(diagonal[Y]),
                math.sqrt(diagonal[THETA]))

    def calibration_state(self):
        if self.compass_sign is None:
            progress = min(1.0, self._calibration_turn / CALIBRATION_TURN_RAD)
            return f'compass=calibrating({progress * 100:.0f}%)'
        return (f'compass=sign{self.compass_sign:+.0f} '
                f'offset={math.degrees(self.compass_offset):+.1f}deg '
                f'gyro_bias={self.gyro_bias:+.4f}rad/s')

    def _predict_velocity(self, dt):
        tracking = COMMAND_TRACKING if self.command is not None else 0.0
        command_v, command_omega = self.command if self.command else (0.0, 0.0)

        jacobian = np.eye(STATE_SIZE)
        jacobian[V, V] = 1.0 - tracking
        jacobian[OMEGA, OMEGA] = 1.0 - tracking
        self.state[V] += tracking * (command_v - self.state[V])
        self.state[OMEGA] += tracking * (command_omega - self.state[OMEGA])

        process_noise = np.zeros((STATE_SIZE, STATE_SIZE))
        process_noise[V, V] = PROCESS_STD[V] ** 2 * dt
        process_noise[OMEGA, OMEGA] = PROCESS_STD[OMEGA] ** 2 * dt
        process_noise[GYRO_BIAS, GYRO_BIAS] = GYRO_BIAS_WALK_STD ** 2 * dt
        self.covariance = (jacobian @ self.covariance @ jacobian.T
                           + process_noise)
        self.command = None

    def _integrate_pose(self, dt):
        theta = self.state[THETA]
        v = self.state[V]
        omega = self.state[OMEGA]
        middle = theta + 0.5 * omega * dt
        cosine = math.cos(middle)
        sine = math.sin(middle)

        jacobian = np.eye(STATE_SIZE)
        jacobian[X, THETA] = -v * sine * dt
        jacobian[X, V] = cosine * dt
        jacobian[X, OMEGA] = -0.5 * v * sine * dt * dt
        jacobian[Y, THETA] = v * cosine * dt
        jacobian[Y, V] = sine * dt
        jacobian[Y, OMEGA] = 0.5 * v * cosine * dt * dt
        jacobian[THETA, OMEGA] = dt

        self.state[X] += v * cosine * dt
        self.state[Y] += v * sine * dt
        self.state[THETA] = wrap_angle(theta + omega * dt)

        process_noise = np.zeros((STATE_SIZE, STATE_SIZE))
        for index in (X, Y, THETA):
            process_noise[index, index] = PROCESS_STD[index] ** 2 * dt
        self.covariance = (jacobian @ self.covariance @ jacobian.T
                           + process_noise)

    def _update_linear(self, measurement, jacobian, noise, angle_rows=()):
        residual = measurement - jacobian @ self.state
        self._apply_update(residual, jacobian, noise, angle_rows)

    def _apply_update(self, residual, jacobian, noise, angle_rows=()):
        for row in angle_rows:
            residual[row] = wrap_angle(residual[row])

        innovation = jacobian @ self.covariance @ jacobian.T + noise
        try:
            gain = np.linalg.solve(
                innovation.T, (self.covariance @ jacobian.T).T).T
        except np.linalg.LinAlgError:
            return

        self.state = self.state + gain @ residual
        self.state[THETA] = wrap_angle(self.state[THETA])

        spread = np.eye(STATE_SIZE) - gain @ jacobian
        self.covariance = (spread @ self.covariance @ spread.T
                           + gain @ noise @ gain.T)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _update_gyro(self, moving):
        self.stationary = not moving
        if not moving:
            self._update_linear(
                np.zeros(2), _rows(V, OMEGA),
                np.diag([ZUPT_V_STD ** 2, ZUPT_OMEGA_STD ** 2]))

        rate = self.gyro[2]
        if not math.isfinite(rate):
            return

        jacobian = np.zeros((1, STATE_SIZE))
        jacobian[0, OMEGA] = 1.0
        jacobian[0, GYRO_BIAS] = 1.0
        self._update_linear(np.array([rate]), jacobian,
                            np.array([[GYRO_OMEGA_STD ** 2]]))

    def _update_accelerometer(self, moving):
        lateral = self.acceleration[1]
        if not math.isfinite(lateral):
            return
        if not moving:
            if self.accel_bias is None:
                self.accel_bias = lateral
            else:
                self.accel_bias += ACCEL_BIAS_ALPHA * (lateral - self.accel_bias)
            return
        if self.accel_bias is None:
            return
        v, omega = self.state[V], self.state[OMEGA]
        if abs(v) < ACCEL_MIN_SPEED:
            return
        residual = (lateral - self.accel_bias) - v * omega
        if abs(residual) > ACCEL_GATE:
            return
        jacobian = np.zeros((1, STATE_SIZE))
        jacobian[0, V] = omega
        jacobian[0, OMEGA] = v
        self._apply_update(np.array([residual]), jacobian,
                           np.array([[ACCEL_LATERAL_STD ** 2]]))
        self.accel_count += 1

    def _update_compass(self, dt):
        raw = self._compass_angle()
        if raw is None:
            return

        if self.compass_sign is None:
            self._calibrate(raw, self.state[OMEGA] * dt)
            return

        heading = wrap_angle(self.compass_sign * raw + self.compass_offset)
        self.compass_heading = heading

        innovation = wrap_angle(heading - self.state[THETA])
        self.compass_variance += COMPASS_ADAPT_ALPHA * (
            innovation * innovation - self.compass_variance)
        noise = max(COMPASS_MIN_THETA_STD ** 2,
                    self.compass_variance - self.covariance[THETA, THETA])

        self._update_linear(
            np.array([heading]),
            _rows(THETA),
            np.array([[noise]]),
            angle_rows=(0,))

    def _compass_angle(self):
        cx, cy = self.compass[0], self.compass[1]
        if not (math.isfinite(cx) and math.isfinite(cy)):
            return None
        if math.hypot(cx, cy) < COMPASS_MIN_NORM:
            return None
        return math.atan2(cx, cy)

    def _calibrate(self, raw, turn):
        self._calibration_samples.append((raw, self.state[THETA]))
        if len(self._calibration_samples) > CALIBRATION_MAX_SAMPLES:
            self._calibration_samples.pop(0)

        if self._calibration_previous_raw is not None and turn != 0.0:
            delta = wrap_angle(raw - self._calibration_previous_raw)
            self._calibration_correlation += turn * delta
            self._calibration_turn += abs(turn)
        self._calibration_previous_raw = raw

        if self._calibration_turn < CALIBRATION_TURN_RAD:
            return
        if self._calibration_correlation == 0.0:
            return

        sign = 1.0 if self._calibration_correlation > 0.0 else -1.0
        sin_sum = 0.0
        cos_sum = 0.0
        for sample_raw, sample_theta in self._calibration_samples:
            difference = wrap_angle(sample_theta - sign * sample_raw)
            sin_sum += math.sin(difference)
            cos_sum += math.cos(difference)

        self.compass_sign = sign
        self.compass_offset = math.atan2(sin_sum, cos_sum)
        self._calibration_samples.clear()

    def _sync_pose(self):
        self.pose.x = float(self.state[X])
        self.pose.y = float(self.state[Y])
        self.pose.theta = float(self.state[THETA])
        return self.pose


def _rows(*indices):
    jacobian = np.zeros((len(indices), STATE_SIZE))
    for row, index in enumerate(indices):
        jacobian[row, index] = 1.0
    return jacobian


def _as_triple(values):
    try:
        return (float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0, 0.0)


CAMERA_X = 0.14
CAMERA_BAND = (40, 12)
CAMERA_MIN_COLUMNS = 4
CAMERA_MIN_PIXELS = 4
LANDMARK_COLORS = {
    'green': [((40, 100, 30), (80, 255, 255))],
    'blue': [((105, 120, 30), (130, 255, 255))],
    'pink': [((160, 100, 35), (180, 255, 255)), ((0, 100, 35), (4, 255, 255))],
}
LANDMARK_MAX_RANGE = 6.0
LANDMARK_MIN_POINTS = 6
LANDMARK_RADIUS = (0.15, 0.95)
LANDMARK_MAX_RESIDUAL = 0.03
LANDMARK_REGISTER_COUNT = 5
LANDMARK_REGISTER_MAX_STD = 0.05
LANDMARK_ASSOCIATE_M = 0.6
LANDMARK_RANGE_STD = 0.03
LANDMARK_BEARING_STD = 0.01
LANDMARK_GATE = (0.30, 0.12)


def _fit_circle(xs, ys):
    a = np.column_stack((xs, ys, np.ones_like(xs)))
    b = -(xs * xs + ys * ys)
    try:
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -0.5 * solution[0], -0.5 * solution[1]
    squared = cx * cx + cy * cy - solution[2]
    if squared <= 0.0:
        return None
    radius = math.sqrt(squared)
    residual = np.hypot(xs - cx, ys - cy) - radius
    return cx, cy, radius, float(np.sqrt(np.mean(residual * residual)))


class CameraLandmarks:
    def __init__(self):
        self.landmarks = []
        self.last_detections = []
        self.used = 0
        self.rejected = 0

    def detect(self, image, width, height, fov, lidar_points, bearing_offset=0.0):
        if cv2 is None or image is None:
            return []
        pixels = np.frombuffer(image, np.uint8) if isinstance(image, (bytes, bytearray)) else np.asarray(image)
        pixels = pixels.reshape(height, width, -1)[:, :, :3]
        top, bottom = height // 2 - CAMERA_BAND[0], height // 2 - CAMERA_BAND[1]
        hsv = cv2.cvtColor(np.ascontiguousarray(pixels[top:bottom]), cv2.COLOR_BGR2HSV)
        focal = (width / 2.0) / math.tan(fov / 2.0)

        points = _planar_points(lidar_points, bearing_offset)
        if len(points) == 0:
            return []
        from_camera = np.arctan2(points[:, 1], points[:, 0] - CAMERA_X)
        reach = np.hypot(points[:, 0], points[:, 1])

        found = []
        for color, ranges in LANDMARK_COLORS.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, low, high)
            columns = (mask > 0).sum(axis=0) >= CAMERA_MIN_PIXELS
            for first, last in _runs(columns, CAMERA_MIN_COLUMNS):
                if first == 0 or last == width - 1:
                    continue
                left = math.atan((width / 2.0 - first) / focal)
                right = math.atan((width / 2.0 - last - 1) / focal)
                inside = ((from_camera <= left) & (from_camera >= right) &
                          (reach < LANDMARK_MAX_RANGE))
                if int(inside.sum()) < LANDMARK_MIN_POINTS:
                    continue
                selected = points[inside]
                nearest = float(np.min(np.hypot(selected[:, 0], selected[:, 1])))
                selected = selected[np.hypot(selected[:, 0], selected[:, 1]) < nearest + 0.5]
                if len(selected) < LANDMARK_MIN_POINTS:
                    continue
                circle = _fit_circle(selected[:, 0], selected[:, 1])
                if circle is None:
                    continue
                cx, cy, radius, residual = circle
                if not (LANDMARK_RADIUS[0] <= radius <= LANDMARK_RADIUS[1]):
                    continue
                if residual > LANDMARK_MAX_RESIDUAL:
                    continue
                found.append((color, cx, cy, radius))
        self.last_detections = found
        return found

    def associate(self, color, world_x, world_y):
        best, best_distance = None, LANDMARK_ASSOCIATE_M
        for landmark in self.landmarks:
            if landmark['color'] != color:
                continue
            distance = math.hypot(landmark['x'] - world_x, landmark['y'] - world_y)
            if distance < best_distance:
                best, best_distance = landmark, distance
        return best


def _runs(flags, minimum):
    runs, start = [], None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                runs.append((start, index - 1))
            start = None
    return runs


def _planar_points(lidar_points, bearing_offset):
    if isinstance(lidar_points, np.ndarray):
        points = lidar_points[:, :2].astype(float)
    else:
        points = np.array([(p.x, p.y) for p in lidar_points
                           if math.isfinite(p.x) and math.isfinite(p.y)], dtype=float)
    if len(points) == 0:
        return points.reshape(0, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if bearing_offset:
        c, s = math.cos(bearing_offset), math.sin(bearing_offset)
        points = np.column_stack((c * points[:, 0] - s * points[:, 1],
                                  s * points[:, 0] + c * points[:, 1]))
    return points[np.hypot(points[:, 0], points[:, 1]) > 0.1]
