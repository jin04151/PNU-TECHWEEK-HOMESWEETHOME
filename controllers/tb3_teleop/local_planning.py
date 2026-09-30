"""Look-ahead path tracking with a local-costmap avoidance layer."""

import math
import time

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

# Local obstacle handling.
SLOWDOWN_DISTANCE = 0.80
EMERGENCY_FRONT_DISTANCE = 0.30
AVOID_TRIGGER_DISTANCE = 0.55

# Once avoidance starts, keep it active until the obstacle is a little farther
# away than the trigger distance. This hysteresis prevents rapid ON/OFF changes.
AVOID_RELEASE_DISTANCE = 0.70

# Corner/dead-end recovery. If no normal avoidance trajectory is safe,
# rotate in one chosen direction until the front opens again.
RECOVERY_TURN_SPEED = 0.55
RECOVERY_RELEASE_DISTANCE = 0.80
RECOVERY_SIDE_HALF_WIDTH_DEG = 35
RECOVERY_SIDE_SWITCH_MARGIN = 0.10

# Stuck recovery.
# If forward motion is commanded but the estimated position barely changes
# for long enough, assume that the robot is physically stuck on something
# that may not be visible to the LiDAR (for example, a carpet edge).
STUCK_DETECT_TIME = 1.5
STUCK_DISTANCE_THRESHOLD = 0.03
STUCK_MIN_FORWARD_COMMAND = 0.03
STUCK_RECOVERY_TURN_SPEED = 0.60
STUCK_RECOVERY_ANGLE = math.radians(50.0)
STUCK_RECOVERY_TIMEOUT = 1.8

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


# Path-tracking state.
_active_path_id = None
_path_polyline = None
_progress_segment = 0

# Local-avoidance state.
_last_turn_sign = 0
_avoidance_active = False
_recovery_active = False
_recovery_turn_sign = 0

# Progress-based stuck detection state.
_stuck_anchor = None
_stuck_anchor_time = None
_stuck_recovery_active = False
_stuck_recovery_turn_sign = 0
_stuck_recovery_start_theta = None
_stuck_recovery_start_time = None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _reset_tracking():
    """Reset both path-progress and local-avoidance state."""
    global _active_path_id
    global _path_polyline
    global _progress_segment
    global _last_turn_sign
    global _avoidance_active
    global _recovery_active
    global _recovery_turn_sign
    global _stuck_anchor
    global _stuck_anchor_time
    global _stuck_recovery_active
    global _stuck_recovery_turn_sign
    global _stuck_recovery_start_theta
    global _stuck_recovery_start_time

    _active_path_id = None
    _path_polyline = None
    _progress_segment = 0

    _last_turn_sign = 0
    _avoidance_active = False
    _recovery_active = False
    _recovery_turn_sign = 0

    _stuck_anchor = None
    _stuck_anchor_time = None
    _stuck_recovery_active = False
    _stuck_recovery_turn_sign = 0
    _stuck_recovery_start_theta = None
    _stuck_recovery_start_time = None


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

    return (
        ax + t * vx,
        ay + t * vy,
    ), t


def _initialize_or_update_path(pose, path):
    global _active_path_id
    global _path_polyline
    global _progress_segment

    path_id = id(path)

    if path_id != _active_path_id:
        _active_path_id = path_id

        # The global path excludes the current start point, so prepend the
        # robot position when a new path arrives.
        _path_polyline = [
            (pose.x, pose.y)
        ] + list(path)

        _progress_segment = 0


def _lookahead_point(pose, path, lookahead_distance):
    """Project the robot onto the path, then move forward by path distance."""
    global _progress_segment

    _initialize_or_update_path(
        pose,
        path,
    )

    polyline = _path_polyline

    if len(polyline) < 2:
        return path[-1]

    robot_point = (
        pose.x,
        pose.y,
    )

    best_distance = math.inf
    best_segment = _progress_segment
    best_projection = polyline[_progress_segment]

    # Never search behind already-consumed path segments.
    for index in range(
        _progress_segment,
        len(polyline) - 1,
    ):
        projection, _ = _project_point_to_segment(
            robot_point,
            polyline[index],
            polyline[index + 1],
        )

        distance = _distance(
            robot_point,
            projection,
        )

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

        segment_length = _distance(
            current,
            endpoint,
        )

        if (
            segment_length >= remaining
            and segment_length > 1e-12
        ):
            ratio = (
                remaining
                / segment_length
            )

            return (
                current[0]
                + ratio
                * (endpoint[0] - current[0]),

                current[1]
                + ratio
                * (endpoint[1] - current[1]),
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

    # Robot frame:
    # +x = forward
    # +y = left
    x_robot = (
        c * dx
        + s * dy
    )

    y_robot = (
        -s * dx
        + c * dy
    )

    return (
        x_robot,
        y_robot,
    )


def _pure_pursuit_reference(pose, path):
    """Generate the normal path-following command before obstacle avoidance."""
    lookahead = _lookahead_point(
        pose,
        path,
        LOOKAHEAD_DISTANCE,
    )

    x_la, y_la = _world_to_robot_frame(
        pose,
        lookahead,
    )

    distance_sq = (
        x_la * x_la
        + y_la * y_la
    )

    if distance_sq <= 1e-9:
        return (
            0.0,
            0.0,
            lookahead,
        )

    curvature = (
        2.0 * y_la
        / distance_sq
    )

    # Reduce speed on sharp curves.
    curve_factor = (
        1.0
        / (
            1.0
            + 0.75 * abs(curvature)
        )
    )

    linear = _clamp(
        FORWARD_SPEED * curve_factor,
        MIN_FORWARD_SPEED,
        FORWARD_SPEED,
    )

    angular = _clamp(
        linear * curvature,
        -MAX_ANGULAR_SPEED,
        MAX_ANGULAR_SPEED,
    )

    return (
        linear,
        angular,
        lookahead,
    )


def nearest_front_obstacle(
    ranges,
    half_width_degrees=15,
):
    """Return the nearest range around the front direction (index 180)."""
    if not ranges:
        return math.inf

    count = len(ranges)
    front = count // 2

    half_width = max(
        1,
        int(
            round(
                count
                * half_width_degrees
                / 360.0
            )
        ),
    )

    nearest = math.inf

    for offset in range(
        -half_width,
        half_width + 1,
    ):
        value = ranges[
            (front + offset)
            % count
        ]

        if (
            math.isfinite(value)
            and value > 0.0
        ):
            nearest = min(
                nearest,
                value,
            )

    return nearest



def _sector_clearance(ranges, center_index, half_width_degrees):
    """Return conservative clearance in a LiDAR sector."""
    if not ranges:
        return math.inf

    count = len(ranges)
    half_width = max(
        1,
        int(round(count * half_width_degrees / 360.0)),
    )

    values = []

    for offset in range(-half_width, half_width + 1):
        value = ranges[(center_index + offset) % count]

        if math.isfinite(value) and value > 0.0:
            values.append(value)

    if not values:
        return math.inf

    # Use a low percentile instead of the absolute minimum so that a single
    # noisy beam does not flip the recovery direction every control cycle.
    values.sort()
    index = min(len(values) - 1, max(0, int(0.20 * (len(values) - 1))))
    return values[index]


def _side_clearances(ranges):
    """Return (left, right) LiDAR clearances.

    LDS-01 convention used by the starter examples:
      index 180 = front
      index  90 = left
      index 270 = right
    """
    if not ranges:
        return math.inf, math.inf

    count = len(ranges)
    left_center = count // 4
    right_center = (3 * count) // 4

    left = _sector_clearance(
        ranges,
        left_center,
        RECOVERY_SIDE_HALF_WIDTH_DEG,
    )

    right = _sector_clearance(
        ranges,
        right_center,
        RECOVERY_SIDE_HALF_WIDTH_DEG,
    )

    return left, right


def _angle_delta(current, reference):
    """Return the wrapped absolute heading change in radians."""
    delta = current - reference

    while delta > math.pi:
        delta -= 2.0 * math.pi

    while delta < -math.pi:
        delta += 2.0 * math.pi

    return abs(delta)


def _reset_stuck_watch(pose=None, now=None):
    """Restart the progress window used to detect physical sticking."""
    global _stuck_anchor
    global _stuck_anchor_time

    if pose is None:
        _stuck_anchor = None
        _stuck_anchor_time = None
        return

    if now is None:
        now = time.monotonic()

    _stuck_anchor = (
        float(pose.x),
        float(pose.y),
    )
    _stuck_anchor_time = now


def _start_stuck_recovery(pose, ranges, angular_hint):
    """Start a committed in-place turn after no forward progress is detected."""
    global _stuck_recovery_active
    global _stuck_recovery_turn_sign
    global _stuck_recovery_start_theta
    global _stuck_recovery_start_time
    global _avoidance_active
    global _recovery_active
    global _recovery_turn_sign

    _stuck_recovery_active = True
    _stuck_recovery_turn_sign = _choose_recovery_turn(
        ranges,
        angular_hint,
    )
    _stuck_recovery_start_theta = float(pose.theta)
    _stuck_recovery_start_time = time.monotonic()

    # Stuck recovery owns the command while it is active.  Clear the normal
    # avoidance/corner-recovery latch so the two recovery modes do not fight.
    _avoidance_active = False
    _recovery_active = False
    _recovery_turn_sign = 0
    _reset_stuck_watch()


def _continue_stuck_recovery(pose):
    """Return a recovery turn, or None when enough rotation has completed."""
    global _stuck_recovery_active
    global _stuck_recovery_turn_sign
    global _stuck_recovery_start_theta
    global _stuck_recovery_start_time
    global _last_turn_sign

    if not _stuck_recovery_active:
        return None

    now = time.monotonic()
    turned = _angle_delta(
        float(pose.theta),
        _stuck_recovery_start_theta,
    )
    elapsed = now - _stuck_recovery_start_time

    if (
        turned >= STUCK_RECOVERY_ANGLE
        or elapsed >= STUCK_RECOVERY_TIMEOUT
    ):
        _stuck_recovery_active = False
        _last_turn_sign = _stuck_recovery_turn_sign
        _stuck_recovery_turn_sign = 0
        _stuck_recovery_start_theta = None
        _stuck_recovery_start_time = None
        _reset_stuck_watch(pose, now)
        return None

    angular = (
        _stuck_recovery_turn_sign
        * STUCK_RECOVERY_TURN_SPEED
    )
    _last_turn_sign = _stuck_recovery_turn_sign

    return (
        0.0,
        angular,
        'MOVING',
    )


def _apply_stuck_detection(
    pose,
    ranges,
    linear,
    angular,
    state,
):
    """Replace a forward command with a recovery turn when progress stalls."""
    global _stuck_anchor
    global _stuck_anchor_time

    now = time.monotonic()

    # Only judge progress while the controller is actually asking the robot
    # to translate forward.  Intentional in-place turns must not look stuck.
    if (
        state != 'MOVING'
        or linear < STUCK_MIN_FORWARD_COMMAND
    ):
        _reset_stuck_watch(pose, now)
        return linear, angular, state

    if (
        _stuck_anchor is None
        or _stuck_anchor_time is None
    ):
        _reset_stuck_watch(pose, now)
        return linear, angular, state

    moved = math.hypot(
        float(pose.x) - _stuck_anchor[0],
        float(pose.y) - _stuck_anchor[1],
    )

    # Any meaningful progress restarts the observation window.
    if moved >= STUCK_DISTANCE_THRESHOLD:
        _reset_stuck_watch(pose, now)
        return linear, angular, state

    if now - _stuck_anchor_time < STUCK_DETECT_TIME:
        return linear, angular, state

    # A forward command has been active for long enough but the robot has not
    # translated.  LiDAR may still look clear (e.g. carpet/threshold), so use
    # the existing left/right clearance logic and rotate toward the open side.
    _start_stuck_recovery(
        pose,
        ranges,
        angular,
    )

    return (
        0.0,
        _stuck_recovery_turn_sign * STUCK_RECOVERY_TURN_SPEED,
        'MOVING',
    )


def _choose_recovery_turn(ranges, angular_ref):
    """Choose one recovery direction and keep it until recovery ends."""
    left_clearance, right_clearance = _side_clearances(ranges)

    # If one side is clearly more open, turn toward it.
    if math.isinf(left_clearance) and not math.isinf(right_clearance):
        return 1

    if math.isinf(right_clearance) and not math.isinf(left_clearance):
        return -1

    if not (math.isinf(left_clearance) and math.isinf(right_clearance)):
        if left_clearance > right_clearance + RECOVERY_SIDE_SWITCH_MARGIN:
            return 1

        if right_clearance > left_clearance + RECOVERY_SIDE_SWITCH_MARGIN:
            return -1

    # Similar clearance: prefer the direction already suggested by the
    # global-path Pure-Pursuit command.
    if angular_ref > 0.08:
        return 1

    if angular_ref < -0.08:
        return -1

    # If avoidance was already turning consistently, preserve that direction.
    if _last_turn_sign:
        return _last_turn_sign

    return 1


def _recovery_command(ranges, linear_ref, angular_ref, front_distance):
    """Rotate out of a corner/dead-end without choosing a new side each loop."""
    global _recovery_active
    global _recovery_turn_sign
    global _avoidance_active
    global _last_turn_sign

    # Exit recovery only after the normal path is safe again and the front
    # has opened by a comfortable margin.
    reference_safe, _ = _reference_trajectory_safe(
        linear_ref,
        angular_ref,
    )

    if (
        _recovery_active
        and reference_safe
        and front_distance >= RECOVERY_RELEASE_DISTANCE
    ):
        _recovery_active = False
        _recovery_turn_sign = 0
        _avoidance_active = False
        _last_turn_sign = 0

        return linear_ref, angular_ref, 'MOVING'

    if not _recovery_active:
        _recovery_active = True
        _recovery_turn_sign = _choose_recovery_turn(
            ranges,
            angular_ref,
        )

    preferred_angular = (
        _recovery_turn_sign
        * RECOVERY_TURN_SPEED
    )

    preferred_safe, _ = _local_costmap.trajectory_score(
        0.0,
        preferred_angular,
        horizon=0.8,
    )

    if preferred_safe:
        _last_turn_sign = _recovery_turn_sign
        return 0.0, preferred_angular, 'MOVING'

    # Only switch direction when the originally committed in-place turn is
    # itself unsafe. This prevents LEFT/RIGHT/LEFT/RIGHT oscillation.
    opposite_sign = -_recovery_turn_sign
    opposite_angular = opposite_sign * RECOVERY_TURN_SPEED

    opposite_safe, _ = _local_costmap.trajectory_score(
        0.0,
        opposite_angular,
        horizon=0.8,
    )

    if opposite_safe:
        _recovery_turn_sign = opposite_sign
        _last_turn_sign = opposite_sign
        return 0.0, opposite_angular, 'MOVING'

    # There is genuinely no locally safe rotation.
    return 0.0, 0.0, 'BLOCKED'

def _candidate_commands(
    linear_ref,
    angular_ref,
    front_distance,
):
    """Generate motion candidates only for the avoidance layer."""
    if (
        front_distance
        < EMERGENCY_FRONT_DISTANCE
    ):
        # Too close in front:
        # do not move forward until a safe turn is found.
        linear_values = [
            0.0,
        ]

    else:
        linear_values = [
            linear_ref,
            max(
                MIN_FORWARD_SPEED,
                0.65 * linear_ref,
            ),
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
            angular = _clamp(
                angular,
                -MAX_ANGULAR_SPEED,
                MAX_ANGULAR_SPEED,
            )

            key = (
                round(linear, 4),
                round(angular, 4),
            )

            if key not in seen:
                seen.add(key)

                unique.append(
                    (
                        linear,
                        angular,
                    )
                )

    return unique


def _select_local_command(
    linear_ref,
    angular_ref,
    front_distance,
    ranges,
):
    """Choose an avoidance command around the pure-pursuit reference."""
    global _last_turn_sign

    best = None
    best_score = math.inf

    for (
        linear,
        angular,
    ) in _candidate_commands(
        linear_ref,
        angular_ref,
        front_distance,
    ):

        # A complete stop is not a valid MOVING avoidance command.
        # If every actually moving candidate is unsafe, let ``best`` remain
        # None so that the committed corner-recovery turn can take over.
        if (
            abs(linear) < 1e-9
            and abs(angular) < 1e-9
        ):
            continue

        # If something is clearly in front,
        # do not keep creeping nearly straight toward it.
        if (
            front_distance
            < AVOID_TRIGGER_DISTANCE
            and abs(angular) < 0.15
        ):
            continue

        (
            safe,
            obstacle_cost,
        ) = _local_costmap.trajectory_score(
            linear,
            angular,
            horizon=1.5,
        )

        if not safe:
            continue

        # Prefer commands close to the pure-pursuit reference.
        tracking_cost = (
            abs(
                angular
                - angular_ref
            )
            / MAX_ANGULAR_SPEED
        )

        # Prefer keeping forward speed when safe.
        speed_cost = (
            FORWARD_SPEED
            - linear
        ) / FORWARD_SPEED

        turn_sign = 0

        if angular > 0.08:
            turn_sign = 1

        elif angular < -0.08:
            turn_sign = -1

        # Avoid rapid LEFT -> RIGHT -> LEFT switching.
        switch_cost = 0.0

        if (
            _last_turn_sign
            and turn_sign
            and turn_sign
            != _last_turn_sign
        ):
            switch_cost = 1.0

        score = (
            TRACKING_WEIGHT
            * tracking_cost

            + OBSTACLE_WEIGHT
            * obstacle_cost

            + SPEED_WEIGHT
            * speed_cost

            + TURN_SWITCH_WEIGHT
            * switch_cost
        )

        if score < best_score:
            best_score = score

            best = (
                linear,
                angular,
                turn_sign,
            )

    if best is None:
        # Normal local trajectories are all blocked. Instead of stopping and
        # re-deciding LEFT/RIGHT every cycle, enter a committed corner
        # recovery turn.
        return _recovery_command(
            ranges,
            linear_ref,
            angular_ref,
            front_distance,
        )

    (
        linear,
        angular,
        turn_sign,
    ) = best

    if turn_sign:
        _last_turn_sign = turn_sign

    return (
        linear,
        angular,
        'MOVING',
    )


def _reference_trajectory_safe(
    linear_ref,
    angular_ref,
):
    """Check whether normal pure-pursuit motion is locally collision-free."""
    safe, obstacle_cost = (
        _local_costmap
        .trajectory_score(
            linear_ref,
            angular_ref,
            horizon=1.5,
        )
    )

    return (
        bool(safe),
        obstacle_cost,
    )


def wheel_speeds(
    linear,
    angular,
):
    left = (
        linear
        - angular
        * WHEEL_SEPARATION
        / 2.0
    ) / WHEEL_RADIUS

    right = (
        linear
        + angular
        * WHEEL_SEPARATION
        / 2.0
    ) / WHEEL_RADIUS

    return (
        left,
        right,
    )


def command(
    pose,
    path,
    ranges,
):
    """Return (linear m/s, angular rad/s, state).

    Normal case:
        Pure Pursuit directly follows the global path.

    Avoidance case:
        The local-costmap trajectory critic is activated only when
        the reference trajectory is unsafe or an obstacle is close ahead.
    """
    global _avoidance_active
    global _last_turn_sign
    global _stuck_recovery_active

    # No path exists.
    if path is None:
        _reset_tracking()

        return (
            0.0,
            0.0,
            'BLOCKED',
        )

    # Empty path means the planner says the goal is already reached.
    if not path:
        _reset_tracking()

        return (
            0.0,
            0.0,
            'ARRIVED',
        )

    goal = path[-1]

    if (
        math.hypot(
            goal[0] - pose.x,
            goal[1] - pose.y,
        )
        <= GOAL_REACHED_DISTANCE
    ):
        _reset_tracking()

        return (
            0.0,
            0.0,
            'ARRIVED',
        )

    # ------------------------------------------------------------
    # 0. Progress-based stuck recovery.
    #
    # If the previous forward commands produced almost no position change,
    # temporarily rotate in place before resuming normal path tracking.
    # ------------------------------------------------------------

    if _stuck_recovery_active:
        stuck_command = _continue_stuck_recovery(
            pose
        )

        if stuck_command is not None:
            return stuck_command

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
        recovery_command = _recovery_command(
            ranges,
            linear_ref,
            angular_ref,
            front_distance,
        )

        return _apply_stuck_detection(
            pose,
            ranges,
            *recovery_command,
        )

    # If an obstacle is extremely close in front, commit to one in-place
    # recovery turn immediately instead of re-choosing left/right every loop.
    if front_distance < EMERGENCY_FRONT_DISTANCE:
        recovery_command = _recovery_command(
            ranges,
            linear_ref,
            angular_ref,
            front_distance,
        )

        return _apply_stuck_detection(
            pose,
            ranges,
            *recovery_command,
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

        return _apply_stuck_detection(
            pose,
            ranges,
            linear_ref,
            angular_ref,
            'MOVING',
        )

    # ------------------------------------------------------------
    # 8. Only while avoidance is active, evaluate alternative
    #    local trajectories and select the safest useful command.
    # ------------------------------------------------------------

    local_command = _select_local_command(
        linear_ref,
        angular_ref,
        front_distance,
        ranges,
    )

    return _apply_stuck_detection(
        pose,
        ranges,
        *local_command,
    )


