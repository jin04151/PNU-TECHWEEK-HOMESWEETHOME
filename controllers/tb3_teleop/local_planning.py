"""Look-ahead path tracking with a local-costmap avoidance layer."""

import math

from local_costmap import LocalCostmap


# Parameters provided by the TECH WEEK notebook for TurtleBot3 Burger.
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160
ROBOT_RADIUS = 0.105

# The values below are controller tuning values, not fixed by the lecture.
FORWARD_SPEED = 0.12
MIN_FORWARD_SPEED = 0.04
MAX_ANGULAR_SPEED = 0.80
GOAL_REACHED_DISTANCE = 0.15
LOOKAHEAD_DISTANCE = 0.40
SLOWDOWN_DISTANCE = 0.80
EMERGENCY_FRONT_DISTANCE = 0.30
AVOID_TRIGGER_DISTANCE = 0.55

# Lightweight critic weights for the local avoidance layer.
TRACKING_WEIGHT = 1.7
OBSTACLE_WEIGHT = 1.4
SPEED_WEIGHT = 0.45
TURN_SWITCH_WEIGHT = 0.25

_local_costmap = LocalCostmap(
    window_size_m=2.4,
    resolution=0.05,
    robot_radius=ROBOT_RADIUS,
    safety_clearance=0.10,
)

_active_path_id = None
_path_polyline = None
_progress_segment = 0
_last_turn_sign = 0


def _clamp(value, low, high):
    return max(low, min(high, value))


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _reset_tracking():
    global _active_path_id, _path_polyline, _progress_segment
    _active_path_id = None
    _path_polyline = None
    _progress_segment = 0


def _project_point_to_segment(point, a, b):
    ax, ay = a
    bx, by = b
    px, py = point

    vx = bx - ax
    vy = by - ay
    length_sq = vx * vx + vy * vy

    if length_sq <= 1e-12:
        return a, 0.0

    t = ((px - ax) * vx + (py - ay) * vy) / length_sq
    t = _clamp(t, 0.0, 1.0)
    return (ax + t * vx, ay + t * vy), t


def _initialize_or_update_path(pose, path):
    global _active_path_id, _path_polyline, _progress_segment

    path_id = id(path)
    if path_id != _active_path_id:
        _active_path_id = path_id
        _path_polyline = [(pose.x, pose.y)] + list(path)
        _progress_segment = 0


def _lookahead_point(pose, path, lookahead_distance):
    """Project the robot onto the path, then move forward by path distance."""
    global _progress_segment

    _initialize_or_update_path(pose, path)
    polyline = _path_polyline

    if len(polyline) < 2:
        return path[-1]

    robot_point = (pose.x, pose.y)
    best_distance = math.inf
    best_segment = _progress_segment
    best_projection = polyline[_progress_segment]

    # Never search behind already-consumed path segments.
    for index in range(_progress_segment, len(polyline) - 1):
        projection, _ = _project_point_to_segment(
            robot_point,
            polyline[index],
            polyline[index + 1],
        )
        distance = _distance(robot_point, projection)

        if distance < best_distance:
            best_distance = distance
            best_segment = index
            best_projection = projection

    _progress_segment = best_segment

    remaining = lookahead_distance
    current = best_projection
    segment_index = best_segment

    while segment_index < len(polyline) - 1:
        endpoint = polyline[segment_index + 1]
        segment_length = _distance(current, endpoint)

        if segment_length >= remaining and segment_length > 1e-12:
            ratio = remaining / segment_length
            return (
                current[0] + ratio * (endpoint[0] - current[0]),
                current[1] + ratio * (endpoint[1] - current[1]),
            )

        remaining -= segment_length
        segment_index += 1
        current = polyline[segment_index]

    return polyline[-1]


def _world_to_robot_frame(pose, point):
    dx = point[0] - pose.x
    dy = point[1] - pose.y
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)

    # +x forward, +y left
    x_robot = c * dx + s * dy
    y_robot = -s * dx + c * dy
    return x_robot, y_robot


def _pure_pursuit_reference(pose, path):
    lookahead = _lookahead_point(pose, path, LOOKAHEAD_DISTANCE)
    x_la, y_la = _world_to_robot_frame(pose, lookahead)
    distance_sq = x_la * x_la + y_la * y_la

    if distance_sq <= 1e-9:
        return 0.0, 0.0, lookahead

    curvature = 2.0 * y_la / distance_sq

    # Reduce speed on sharp curves.
    curve_factor = 1.0 / (1.0 + 0.75 * abs(curvature))
    linear = _clamp(FORWARD_SPEED * curve_factor, MIN_FORWARD_SPEED, FORWARD_SPEED)
    angular = _clamp(linear * curvature, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

    return linear, angular, lookahead


def nearest_front_obstacle(ranges, half_width_degrees=15):
    """Return the nearest range around the front direction (index 180)."""
    if not ranges:
        return math.inf

    count = len(ranges)
    front = count // 2
    half_width = max(1, int(round(count * half_width_degrees / 360.0)))
    nearest = math.inf

    for offset in range(-half_width, half_width + 1):
        value = ranges[(front + offset) % count]
        if math.isfinite(value) and value > 0.0:
            nearest = min(nearest, value)

    return nearest


def _candidate_commands(linear_ref, angular_ref, front_distance):
    if front_distance < EMERGENCY_FRONT_DISTANCE:
        linear_values = [0.0]
    else:
        linear_values = [
            linear_ref,
            max(MIN_FORWARD_SPEED, 0.65 * linear_ref),
            0.0,
        ]

    angular_values = [
        angular_ref,
        angular_ref - 0.30,
        angular_ref + 0.30,
        -0.55,
        0.55,
        -MAX_ANGULAR_SPEED,
        MAX_ANGULAR_SPEED,
        0.0,
    ]

    unique = []
    seen = set()

    for linear in linear_values:
        for angular in angular_values:
            angular = _clamp(angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
            key = (round(linear, 4), round(angular, 4))
            if key not in seen:
                seen.add(key)
                unique.append((linear, angular))

    return unique


def _select_local_command(linear_ref, angular_ref, front_distance):
    """Critic-based avoidance layer over the pure-pursuit reference command."""
    global _last_turn_sign

    best = None
    best_score = math.inf

    for linear, angular in _candidate_commands(linear_ref, angular_ref, front_distance):
        # If something is clearly in front, do not keep creeping straight at it.
        # The avoidance layer must introduce a meaningful turn.
        if (
            front_distance < AVOID_TRIGGER_DISTANCE
            and abs(angular) < 0.15
        ):
            continue

        safe, obstacle_cost = _local_costmap.trajectory_score(
            linear, angular, horizon=1.5
        )
        if not safe:
            continue

        tracking_cost = abs(angular - angular_ref) / MAX_ANGULAR_SPEED
        speed_cost = (FORWARD_SPEED - linear) / FORWARD_SPEED

        turn_sign = 0
        if angular > 0.08:
            turn_sign = 1
        elif angular < -0.08:
            turn_sign = -1

        switch_cost = 0.0
        if _last_turn_sign and turn_sign and turn_sign != _last_turn_sign:
            switch_cost = 1.0

        score = (
            TRACKING_WEIGHT * tracking_cost
            + OBSTACLE_WEIGHT * obstacle_cost
            + SPEED_WEIGHT * speed_cost
            + TURN_SWITCH_WEIGHT * switch_cost
        )

        if score < best_score:
            best_score = score
            best = (linear, angular, turn_sign)

    if best is None:
        return 0.0, 0.0, 'BLOCKED'

    linear, angular, turn_sign = best
    if turn_sign:
        _last_turn_sign = turn_sign

    return linear, angular, 'MOVING'


def wheel_speeds(linear, angular):
    left = (linear - angular * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    right = (linear + angular * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    return left, right


def command(pose, path, ranges):
    """Return (linear m/s, angular rad/s, state)."""
    if path is None:
        _reset_tracking()
        return 0.0, 0.0, 'BLOCKED'

    if not path:
        _reset_tracking()
        return 0.0, 0.0, 'ARRIVED'

    goal = path[-1]
    if math.hypot(goal[0] - pose.x, goal[1] - pose.y) <= GOAL_REACHED_DISTANCE:
        _reset_tracking()
        return 0.0, 0.0, 'ARRIVED'

    _local_costmap.update(ranges)

    linear_ref, angular_ref, _ = _pure_pursuit_reference(pose, path)
    front_distance = nearest_front_obstacle(ranges)

    if front_distance < SLOWDOWN_DISTANCE:
        scale = (front_distance - EMERGENCY_FRONT_DISTANCE) / (
            SLOWDOWN_DISTANCE - EMERGENCY_FRONT_DISTANCE
        )
        scale = _clamp(scale, 0.0, 1.0)
        linear_ref = max(MIN_FORWARD_SPEED, linear_ref * max(0.35, scale))

<<<<<<< HEAD
    return _select_local_command(linear_ref, angular_ref, front_distance)
=======
    # ------------------------------------------------------------
    # 1. Update the rolling local costmap from the current LiDAR.
    # ------------------------------------------------------------

    _local_costmap.update(
        ranges
    )

    # ------------------------------------------------------------
    # 2. Normal reference command:
    #    global path -> look-ahead -> Pure Pursuit.
    # ------------------------------------------------------------

    (
        linear_ref,
    angular_ref,
        _,
    ) = _pure_pursuit_reference(
        pose,
        path,
    )

# ------------------------------------------------------------
# Pure Pursuit가 goal이 아직 먼데도 (0, 0)을 반환하는 경우
# path tracking 상태를 다시 잡고 reference를 재계산한다.
# ------------------------------------------------------------
    if (
        abs(linear_ref) < 1e-9
        and abs(angular_ref) < 1e-9
    ):
        _reset_tracking()

        (
            linear_ref,
            angular_ref,
            _,
        ) = _pure_pursuit_reference(
            pose,
            path,
        )

    front_distance = (
        nearest_front_obstacle(
            ranges
        )
    )
    # ------------------------------------------------------------
    # 3. Slow down before the obstacle becomes close enough to
    #    require active avoidance.
    # ------------------------------------------------------------

    if (
        linear_ref > 0.0
        and front_distance
        < SLOWDOWN_DISTANCE
    ):
        scale = (
            front_distance
            - EMERGENCY_FRONT_DISTANCE
        ) / (
            SLOWDOWN_DISTANCE
            - EMERGENCY_FRONT_DISTANCE
        )

        scale = _clamp(
            scale,
            0.0,
            1.0,
        )

        linear_ref = max(
            MIN_FORWARD_SPEED,
            linear_ref
            * max(
                0.35,
                scale,
            ),
        )

    # ------------------------------------------------------------
    # 4. Check the Pure-Pursuit command itself.
    #
    #    This is the key change:
    #    we no longer run the avoidance critic on every cycle.
    # ------------------------------------------------------------

    (
        reference_safe,
        _,
    ) = _reference_trajectory_safe(
        linear_ref,
        angular_ref,
    )

    # ------------------------------------------------------------
    # 5. Corner/dead-end recovery.
    #
    #    Once recovery starts, keep the chosen turn direction instead of
    #    choosing LEFT/RIGHT again on every control cycle.
    # ------------------------------------------------------------

    if _recovery_active:
        return _recovery_command(
            ranges,
            linear_ref,
            angular_ref,
            front_distance,
        )

    # If an obstacle is extremely close in front, commit to one in-place
    # recovery turn immediately instead of re-choosing left/right every loop.
    if front_distance < EMERGENCY_FRONT_DISTANCE:
        return _recovery_command(
            ranges,
            linear_ref,
            angular_ref,
            front_distance,
        )

    # ------------------------------------------------------------
    # 6. Avoidance activation with hysteresis.
    #
    #    Enter:
    #      - obstacle closer than AVOID_TRIGGER_DISTANCE, or
    #      - the Pure-Pursuit trajectory is not safe.
    #
    #    Exit:
    #      - Pure-Pursuit trajectory is safe, and
    #      - the front obstacle is farther than AVOID_RELEASE_DISTANCE.
    # ------------------------------------------------------------

    avoidance_required = (
        front_distance
        < AVOID_TRIGGER_DISTANCE
        or not reference_safe
    )

    if avoidance_required:
        _avoidance_active = True

    elif (
        _avoidance_active
        and reference_safe
        and front_distance
        >= AVOID_RELEASE_DISTANCE
    ):
        _avoidance_active = False
        _last_turn_sign = 0

    # ------------------------------------------------------------
    # 7. Normal driving:
    #    if the global-path command is safe, use it as-is.
    #
    #    This prevents the local critic from randomly replacing
    #    a good straight/right/left Pure-Pursuit command.
    # ------------------------------------------------------------

    if not _avoidance_active:
        _last_turn_sign = 0

        return (
            linear_ref,
            angular_ref,
            'MOVING',
        )

    # ------------------------------------------------------------
    # 8. Only while avoidance is active, evaluate alternative
    #    local trajectories and select the safest useful command.
    # ------------------------------------------------------------

    return _select_local_command(
        linear_ref,
        angular_ref,
        front_distance,
        ranges,
    )

>>>>>>> 8a75a071f4f0c6d19ad23ef3f23dda5f4a7e40b7
