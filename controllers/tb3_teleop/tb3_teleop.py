"""Webots entry point: autonomous Search-and-Rescue controller."""

import math
import os
from datetime import datetime
from pathlib import Path

from controller import Robot

import exploration
import global_planning
import local_planning
import perception

from global_costmap import (
    build_global_costmap,
    INSCRIBED_COST,
)

from localization import Localization, Pose
from mapping import OccupancyGrid
from recovery import Recovery
# TODO(통합): mission_manager 구현 후 아래 임무 관련 코드와 함께 복원.
# from mission_manager import MissionManager


# ------------------------------------------------------------
# Team integration configuration
# ------------------------------------------------------------

# TODO(통합):
# 제공된 월드 시작 자세가 필요하면 Pose(x, y, theta)로 지정한다.
START_POSE = None


# ------------------------------------------------------------
# Perception configuration
# ------------------------------------------------------------

PERCEPTION_CALIBRATION = None
PERCEPTION_ASSOCIATION = None
# 동일 사과의 재관측을 연결하는 거리(m). 실제 위치 오차에 따라 조정할 초기값.
TARGET_MATCH_DISTANCE_M = 0.3

# 카메라 timestamp와 현재 simulation time의 관계가 확인되면 True.
CAMERA_TIME_CONFIRMED = False


# ------------------------------------------------------------
# Mission configuration
# ------------------------------------------------------------

# 서로 다른 빨간 사과 2개를 구출한 뒤 시작점으로 복귀.
EVENT_TARGET_COUNT = 2

# 로봇 중심과 사과 추정 위치 사이의 구출 판정 거리(m).
# TODO(통합): 이 거리 이내에서 정지했는지 확인하는 임무 로직에 연결한다.
RESCUE_DISTANCE_M = 0.5


# ------------------------------------------------------------
# Global Planning configuration
# ------------------------------------------------------------

# Global Costmap은 every loop마다 다시 만들 필요가 없으므로
# 일정 주기로 갱신한다.
GLOBAL_COSTMAP_PERIOD = 0.25

# Robot radius 바깥에 둘 inflation 영역.
# 튜닝값.
GLOBAL_SAFETY_CLEARANCE_M = 0.12

# Goal 위치가 이 이상 변하면 새로운 goal로 취급.
GOAL_CHANGE_THRESHOLD_M = 0.05

# A* 실패 후 재시도 간격.
REPLAN_RETRY_INTERVAL = 0.5

# Path가 잠깐 막혔다고 바로 Global Replan하지 않는다.
# Local Planner가 먼저 회피하고,
# 이 시간 이상 계속 막힐 때 Global A*를 다시 실행한다.
PATH_BLOCK_PERSISTENCE = 0.75


def _goal_changed(goal, planned_goal):
    """Goal이 의미 있게 변경됐는지 검사한다."""

    if planned_goal is None:
        return True

    return math.hypot(
        goal[0] - planned_goal[0],
        goal[1] - planned_goal[1],
    ) > GOAL_CHANGE_THRESHOLD_M


def main():

    # ========================================================
    # Webots
    # ========================================================

    robot = Robot()

    # 제공 world의 basicTimeStep을 그대로 사용한다.
    timestep = int(
        robot.getBasicTimeStep()
    )


    # ========================================================
    # Motor
    # ========================================================

    left_motor = robot.getDevice(
        'left wheel motor'
    )

    right_motor = robot.getDevice(
        'right wheel motor'
    )

    for motor in (
        left_motor,
        right_motor,
    ):
        motor.setPosition(
            float('inf')
        )

        motor.setVelocity(
            0.0
        )


    # ========================================================
    # Wheel Encoder
    # ========================================================

    # Starter repo 방식:
    # Motor에 연결된 PositionSensor를 가져온다.

    left_encoder = (
        left_motor.getPositionSensor()
    )

    right_encoder = (
        right_motor.getPositionSensor()
    )

    left_encoder.enable(
        timestep
    )

    right_encoder.enable(
        timestep
    )


    # ========================================================
    # LiDAR
    # ========================================================

    lidar = robot.getDevice(
        'LDS-01'
    )

    lidar.enable(
        timestep
    )

    # Mapping / Perception에서 PointCloud 사용 가능.
    lidar.enablePointCloud()


    # ========================================================
    # Camera
    # ========================================================

    camera = robot.getDevice(
        'camera'
    )

    camera.enable(
        timestep
    )


    # ========================================================
    # IMU-related sensors
    # ========================================================

    gyro = robot.getDevice(
        'gyro'
    )

    accelerometer = robot.getDevice(
        'accelerometer'
    )

    compass = robot.getDevice(
        'compass'
    )

    for sensor in (
        gyro,
        accelerometer,
        compass,
    ):
        sensor.enable(
            timestep
        )


    # 실제 Webots Motor 제한값.
    left_motor_limit = (
        left_motor.getMaxVelocity()
    )

    right_motor_limit = (
        right_motor.getMaxVelocity()
    )


    # ========================================================
    # A: Localization / Mapping
    # ========================================================

    localization = Localization(
        initial_pose=START_POSE
    )

    grid = OccupancyGrid(
        rows=201,
        cols=201,
    )


    # ========================================================
    # B: Exploration / Perception
    # ========================================================

    selector = (
        exploration.FrontierSelector()
    )

    diagnostics = (
        exploration.Diagnostics()
    )

    registry = (
        perception.TargetRegistry(
            TARGET_MATCH_DISTANCE_M
        )
        if TARGET_MATCH_DISTANCE_M
        is not None
        else None
    )

    # 벽·장애물에 붙어 못 움직일 때 후진 → 회전으로 빠져나온다.
    recovery = Recovery()


    # ========================================================
    # Mission
    # ========================================================

    pose = localization.pose

    # mission_manager = None

    # 이전 loop의 Local Planner 상태.
    # Mission FSM에서 ARRIVED 여부를 판단할 때 사용.
    # last_local_state = 'WAIT_GOAL'


    # ========================================================
    # Log / Map
    # ========================================================

    controller_dir = (
        Path(__file__).resolve().parent
    )

    maps_dir = (
        controller_dir / 'maps'
    )

    logs_dir = (
        controller_dir / 'logs'
    )

    maps_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_path = (
        maps_dir / 'map.png'
    )

    log_name = (
        f"run_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}_"
        f"{os.getpid()}.log"
    )

    log_path = (
        logs_dir / log_name
    )

    next_map_time = 5.0
    next_log_time = 0.0


    # ========================================================
    # C: Global Costmap state
    # ========================================================

    global_costmap_data = None

    # B의 기존 interface를 위해
    # bool safe_grid도 costmap에서 만들어준다.
    safe_grid = None

    last_costmap_time = (
        -math.inf
    )


    # ========================================================
    # C: Global Planning / Replanning state
    # ========================================================

    current_path = None

    planned_goal = None

    last_plan_time = (
        -math.inf
    )

    # Path가 언제부터 막혀 있었는지.
    path_blocked_since = None


    # ========================================================
    # Main loop
    # ========================================================

    with log_path.open(
        'w',
        encoding='utf-8',
        buffering=1,
    ) as log_file:

        def write_log(message):
            """콘솔과 파일에 동일한 로그를 기록한다."""

            print(
                message,
                flush=True,
            )

            print(
                message,
                file=log_file,
                flush=True,
            )


        try:

            while (
                robot.step(timestep) != -1
            ):

                now = robot.getTime()


                # ====================================================
                # 0. Sensor input
                # ====================================================

                # Local Planner는 RangeImage 사용.
                lidar_ranges = (
                    lidar.getRangeImage()
                    or []
                )

                # Mapping / Perception은 기존처럼 PointCloud 사용.
                lidar_points = (
                    lidar.getPointCloud()
                    or []
                )

                camera_image = (
                    camera.getImage()
                )

                gyro_values = (
                    gyro.getValues()
                )

                acceleration = (
                    accelerometer.getValues()
                )

                north = (
                    compass.getValues()
                )

                left_angle = (
                    left_encoder.getValue()
                )

                right_angle = (
                    right_encoder.getValue()
                )


                # ====================================================
                # 1. A: Localization
                # ====================================================

                pose = localization.update(
                    left_angle,
                    right_angle,
                    gyro_values,
                    acceleration,
                    north,
                    now,
                )


                # 최초 pose가 생성되면
                # 시작 위치를 MissionManager에 저장.
                # if mission_manager is None and pose is not None:
                #     mission_manager = MissionManager(
                #         start_position=(pose.x, pose.y),
                #         required_target_count=EVENT_TARGET_COUNT,
                #     )


                # ====================================================
                # 2. A: Mapping
                # ====================================================

                grid.update(
                    pose,
                    lidar_points,
                )


                # ====================================================
                # 3. C: Global Costmap
                # ====================================================

                if (
                    global_costmap_data
                    is None
                    or
                    now - last_costmap_time
                    >= GLOBAL_COSTMAP_PERIOD
                ):

                    global_costmap_data = (
                        build_global_costmap(
                            grid.data,
                            grid.resolution,

                            robot_radius=(
                                local_planning.ROBOT_RADIUS
                            ),

                            safety_clearance=(
                                GLOBAL_SAFETY_CLEARANCE_M
                            ),
                        )
                    )


                    # 기존 B FrontierSelector가
                    # bool safe_grid를 사용하므로
                    # costmap에서 생성한다.
                    #
                    # 253 이상:
                    # inscribed / occupied / unknown
                    #
                    # -> 통행 금지

                    safe_grid = (
                        global_costmap_data
                        >= int(INSCRIBED_COST)
                    )

                    last_costmap_time = (
                        now
                    )


                # ====================================================
                # 4. B: Perception
                # ====================================================

                image = (
                    perception
                    .camera_image_to_bgr(
                        camera_image,
                        camera.getWidth(),
                        camera.getHeight(),
                    )
                )

                detections = (
                    []
                    if image is None
                    else
                    perception.detect_targets(
                        image
                    )
                )

                observations = []

                located_count = 0

                # MissionManager에 넘길
                # 현재 발견된 target 위치.
                target_goal = None


                if (
                    PERCEPTION_CALIBRATION
                    is None
                    or
                    PERCEPTION_ASSOCIATION
                    is None
                ):

                    perception_state = (
                        'WAIT_CALIBRATION'
                    )


                elif (
                    not CAMERA_TIME_CONFIRMED
                ):

                    perception_state = (
                        'WAIT_CAMERA_TIME'
                    )


                else:

                    timing = (
                        perception
                        .ObservationTimes(
                            now,
                            now,

                            tuple(
                                getattr(
                                    point,
                                    'time',
                                    math.nan,
                                )
                                for point
                                in lidar_points
                            ),
                        )
                    )


                    observations = [

                        (
                            detection,

                            perception.locate_target(
                                detection,
                                lidar_points,
                                pose,

                                PERCEPTION_CALIBRATION,

                                association=(
                                    PERCEPTION_ASSOCIATION
                                ),

                                timing=timing,
                            ),
                        )

                        for detection
                        in detections
                    ]


                    located_positions = [

                        position

                        for _, position
                        in observations

                        if position
                        is not None
                    ]


                    located_count = len(
                        located_positions
                    )


                    # 현재는 첫 번째 유효 target을
                    # MissionManager에 전달.
                    #
                    # B에서 target 우선순위를
                    # 구현하면 이 부분만 교체하면 됨.
                    target_goal = (

                        located_positions[0]

                        if located_positions

                        else None
                    )


                    perception_state = (

                        'READY'

                        if registry
                        is not None

                        else
                        'WAIT_REGISTRY_CONFIG'
                    )


                    if registry is not None:

                        registry.update(
                            observations,
                            now,
                        )

                # GCS 지도에 발견한 사과를 표시한다(mapping.py 가 같이 보낸다).
                grid.gcs_targets = registry.targets if registry is not None else ()


                # ====================================================
                # 5. B: Frontier Exploration
                # ====================================================

                exploration_goal = (
                    selector.choose(
                        grid,
                        safe_grid,
                        pose,
                        now,

                        allow_switch=False,
                    )
                )


                # ====================================================
                # 6. Mission FSM
                # ====================================================

                # arrived = last_local_state == 'ARRIVED'
                # if mission_manager is None:
                #     goal = exploration_goal
                # else:
                #     # 탐색 완료 판정 연결 후 set_search_complete() 호출.
                #     goal = mission_manager.update(
                #         exploration_goal=exploration_goal,
                #         detected_target_goal=target_goal,
                #         arrived=arrived,
                #     )

                # 임무 관리자 연결 전에는 Frontier 탐색 목표로 주행한다.
                goal = exploration_goal


                # ====================================================
                # 7. C: Conditional Global Replanning
                # ====================================================

                if goal is None:

                    current_path = None
                    planned_goal = None

                    path_blocked_since = (
                        None
                    )


                else:

                    # -----------------------------------------------
                    # Goal change
                    # -----------------------------------------------

                    goal_changed = (
                        _goal_changed(
                            goal,
                            planned_goal,
                        )
                    )


                    # -----------------------------------------------
                    # Cached path validity
                    # -----------------------------------------------

                    path_blocked = (

                        current_path
                        is not None

                        and
                        global_costmap_data
                        is not None

                        and
                        global_planning
                        .path_is_blocked(
                            grid,
                            global_costmap_data,
                            pose,
                            current_path,
                        )
                    )


                    # -----------------------------------------------
                    # Dynamic obstacle handling
                    # -----------------------------------------------
                    #
                    # Path가 잠깐 막힌 경우:
                    #   Local Planner가 먼저 처리.
                    #
                    # 오래 막힌 경우:
                    #   Global A* replan.

                    if path_blocked:

                        if (
                            path_blocked_since
                            is None
                        ):

                            path_blocked_since = (
                                now
                            )

                    else:

                        path_blocked_since = (
                            None
                        )


                    persistent_block = (

                        path_blocked_since
                        is not None

                        and

                        now - path_blocked_since
                        >= PATH_BLOCK_PERSISTENCE
                    )


                    # -----------------------------------------------
                    # Replan decision
                    # -----------------------------------------------

                    need_replan = (

                        current_path
                        is None

                        or goal_changed

                        or persistent_block
                    )


                    if need_replan:

                        # Goal 변경과 지속적 차단은
                        # 즉시 A*.
                        immediate_replan = (

                            goal_changed

                            or persistent_block
                        )


                        # 이전 A* 실패인 경우
                        # 매 loop가 아니라 일정 시간 후 retry.
                        retry_ready = (

                            now - last_plan_time

                            >= REPLAN_RETRY_INTERVAL
                        )


                        if (
                            immediate_replan
                            or retry_ready
                        ):

                            current_path = (
                                global_planning.plan(
                                    grid,
                                    global_costmap_data,
                                    pose,
                                    goal,
                                )
                            )


                            planned_goal = (
                                float(goal[0]),
                                float(goal[1]),
                            )


                            last_plan_time = (
                                now
                            )


                            path_blocked_since = (
                                None
                            )


                path = current_path


                # ====================================================
                # 8. C: Local Planning
                # ====================================================

                if goal is None:

                    linear = 0.0
                    angular = 0.0
                    state = 'WAIT_GOAL'


                elif path is None:

                    linear = 0.0
                    angular = 0.0
                    state = 'WAIT_PATH'


                else:

                    # 내부에서:
                    #
                    # Look-ahead
                    # Pure Pursuit
                    # Local Costmap
                    # trajectory avoidance
                    #
                    # 를 처리한다.

                    (
                        linear,
                        angular,
                        state,
                    ) = local_planning.command(
                        pose,
                        path,
                        lidar_ranges,
                    )


                # 탐색 전용 실행: C의 도착 결과를 B에 전달하고 경로를 해제한다.
                # TODO(통합): Mission FSM 복원 시 탐색 상태에서만 전달한다.
                if state == 'ARRIVED' and goal is not None:
                    if selector.report_result(goal, 'ARRIVED', now):
                        current_path = None
                        planned_goal = None
                        path_blocked_since = None

                # Recovery: 갇히면 로컬 플래너 명령 대신 후진 → 회전.
                # 끝나면 새 위치에서 다시 계획하고, 같은 목표에서 반복 실패하면 B 에 알린다.
                override = recovery.update(now, pose, lidar_ranges, linear, angular, goal)
                if override is not None:
                    linear, angular = override
                    state = f'RECOVERY_{recovery.state}'
                if recovery.finished:
                    current_path = None
                    planned_goal = None
                    path_blocked_since = None
                    write_log(f't={now:.2f}s recovery finished count={recovery.count} '
                              f'gave_up={recovery.gave_up} goal={goal}')
                    if recovery.gave_up and goal is not None:
                        selector.report_result(goal, 'BLOCKED', now)

                # last_local_state = state


                # ====================================================
                # Diagnostic front distance
                # ====================================================

                front_distance = (
                    local_planning
                    .nearest_front_obstacle(
                        lidar_ranges
                    )
                )


                # ====================================================
                # 9. Motor command
                # ====================================================

                (
                    left_speed,
                    right_speed,
                ) = (
                    local_planning
                    .wheel_speeds(
                        linear,
                        angular,
                    )
                )


                # Webots motor limit 보호.
                left_speed = max(
                    -left_motor_limit,
                    min(
                        left_motor_limit,
                        left_speed,
                    ),
                )

                right_speed = max(
                    -right_motor_limit,
                    min(
                        right_motor_limit,
                        right_speed,
                    ),
                )


                left_motor.setVelocity(
                    left_speed
                )

                right_motor.setVelocity(
                    right_speed
                )


                # ====================================================
                # 10. Diagnostics
                # ====================================================

                diagnostics.update(
                    now,
                    grid,
                    safe_grid,
                    pose,
                    goal,
                    write_log,
                    selector=selector,
                )


                if (
                    now >= next_log_time
                ):

                    distance_text = (

                        'clear'

                        if math.isinf(
                            front_distance
                        )

                        else
                        f'{front_distance:.2f}m'
                    )


                    # mission_state = (
                    #     mission_manager.state.name
                    #     if mission_manager is not None else 'INIT'
                    # )
                    mission_state = 'DISABLED'


                    write_log(

                        f't={now:.2f}s '

                        f'state={state} '

                        f'mission='
                        f'{mission_state} '

                        f'front='
                        f'{distance_text} '

                        f'pose=('
                        f'{pose.x:+.2f}, '
                        f'{pose.y:+.2f}, '
                        f'{math.degrees(pose.theta):+.1f}deg) '

                        f'goal={goal} '

                        f'detections='
                        f'{len(detections)} '

                        f'located='
                        f'{located_count} '

                        f'targets='
                        f'{len(registry.targets) if registry is not None else 0} '

                        f'perception='
                        f'{perception_state}'
                    )


                    next_log_time += (
                        1.0
                    )


                # ====================================================
                # 11. Map save
                # ====================================================

                if (
                    now >= next_map_time
                ):

                    diagnostics.save_png(
                        map_path,
                        grid,
                        pose,
                    )

                    write_log(
                        f't={now:.2f}s '
                        f'map_saved={map_path}'
                    )

                    next_map_time += (
                        5.0
                    )


        finally:

            # ========================================================
            # Final map save
            # ========================================================

            if (
                diagnostics.snapshot
                is not None
            ):

                diagnostics.update(
                    robot.getTime(),
                    grid,
                    safe_grid,
                    pose,
                    None,
                    write_log,
                    selector=selector,
                )

                diagnostics.save_png(
                    map_path,
                    grid,
                    pose,
                )

            else:

                grid.save_png(
                    map_path
                )


            write_log(
                f't={robot.getTime():.2f}s '
                f'map_saved={map_path} final'
            )


if __name__ == '__main__':
    main()
