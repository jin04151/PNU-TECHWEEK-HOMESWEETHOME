"""Webots entry point: autonomous Search-and-Rescue controller.

매 스텝: 센서 → A(위치·지도) → C(costmap) → B(인식·탐사) → C(경로·주행) → recovery → 모터.
"""

import math
import os
from datetime import datetime
from pathlib import Path

from controller import Robot

import exploration
import global_planning
import local_planning
import perception
from global_costmap import build_global_costmap, INSCRIBED_COST
from localization import Localization, Pose
from mapping import OccupancyGrid
from recovery import Recovery


# 통합 설정 ----------------------------------------------------------------
START_POSE = None                  # 월드 시작 자세가 필요하면 Pose(x, y, theta)

PERCEPTION_CALIBRATION = None      # B: 카메라-LiDAR 보정값. 정해지면 사과 위치 계산이 켜진다
PERCEPTION_ASSOCIATION = None
TARGET_MATCH_DISTANCE_M = 0.3
CAMERA_TIME_CONFIRMED = False      # 카메라 timestamp 와 시뮬레이션 시각 관계를 확인했으면 True

EVENT_TARGET_COUNT = 2             # 구조 대상 수(미션 매니저 복원 시 사용)
RESCUE_DISTANCE_M = 0.5

GLOBAL_COSTMAP_PERIOD = 0.25       # s, costmap 갱신 주기
GLOBAL_SAFETY_CLEARANCE_M = 0.12   # 로봇 반경 바깥 여유
GOAL_CHANGE_THRESHOLD_M = 0.05     # 이보다 목표가 바뀌면 새 목표
REPLAN_RETRY_INTERVAL = 0.5        # s, A* 실패 후 재시도 간격
PATH_BLOCK_PERSISTENCE = 0.75      # s, 경로가 이만큼 막혀 있으면 다시 계획

LOG_PERIOD = 1.0
MAP_SAVE_PERIOD = 5.0


def goal_changed(goal, planned_goal):
    return planned_goal is None or math.dist(goal, planned_goal) > GOAL_CHANGE_THRESHOLD_M


def locate_targets(detections, lidar_points, pose, now, registry):
    """(인식 상태, 위치 목록) 을 돌려주고 registry 를 갱신한다. 보정값이 없으면 대기."""
    if PERCEPTION_CALIBRATION is None or PERCEPTION_ASSOCIATION is None:
        return 'WAIT_CALIBRATION', []
    if not CAMERA_TIME_CONFIRMED:
        return 'WAIT_CAMERA_TIME', []
    timing = perception.ObservationTimes(now, now, tuple(getattr(p, 'time', math.nan) for p in lidar_points))
    observations = [(d, perception.locate_target(d, lidar_points, pose, PERCEPTION_CALIBRATION,
                                                 association=PERCEPTION_ASSOCIATION, timing=timing))
                    for d in detections]
    if registry is None:
        return 'WAIT_REGISTRY_CONFIG', [p for _, p in observations if p is not None]
    registry.update(observations, now)
    return 'READY', [p for _, p in observations if p is not None]


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 장치. 나침반·GNSS 는 대회 규정상 쓰지 않는다.
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    for motor in (left_motor, right_motor):
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
    left_encoder, right_encoder = left_motor.getPositionSensor(), right_motor.getPositionSensor()
    lidar, camera = robot.getDevice('LDS-01'), robot.getDevice('camera')
    gyro, accelerometer = robot.getDevice('gyro'), robot.getDevice('accelerometer')
    for device in (left_encoder, right_encoder, lidar, camera, gyro, accelerometer):
        device.enable(timestep)
    lidar.enablePointCloud()
    speed_limit = min(left_motor.getMaxVelocity(), right_motor.getMaxVelocity())

    # 모듈
    localization = Localization(initial_pose=START_POSE)
    grid = OccupancyGrid(rows=201, cols=201, localization=localization)   # 정합 보정이 EKF 로 들어간다
    selector = exploration.FrontierSelector()
    diagnostics = exploration.Diagnostics()
    registry = perception.TargetRegistry(TARGET_MATCH_DISTANCE_M) if TARGET_MATCH_DISTANCE_M else None
    recovery = Recovery()
    pose = localization.pose

    here = Path(__file__).resolve().parent
    (here / 'maps').mkdir(exist_ok=True)
    (here / 'logs').mkdir(exist_ok=True)
    map_path = here / 'maps' / 'map.png'
    log_path = here / 'logs' / f'run_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log'

    costmap, safe_grid, last_costmap_time = None, None, -math.inf
    path, planned_goal, last_plan_time, blocked_since = None, None, -math.inf, None
    next_log_time, next_map_time = 0.0, MAP_SAVE_PERIOD

    def reset_path():
        nonlocal path, planned_goal, blocked_since
        path, planned_goal, blocked_since = None, None, None

    with log_path.open('w', encoding='utf-8', buffering=1) as log_file:
        def write_log(message):
            print(message, flush=True)
            print(message, file=log_file, flush=True)

        try:
            while robot.step(timestep) != -1:
                now = robot.getTime()
                lidar_ranges = lidar.getRangeImage() or []
                lidar_points = lidar.getPointCloud() or []

                # A: 위치·지도
                pose = localization.update(left_encoder.getValue(), right_encoder.getValue(),
                                           gyro.getValues(), accelerometer.getValues(), None, now)
                grid.update(pose, lidar_points)

                # C: global costmap (주기적으로)
                if costmap is None or now - last_costmap_time >= GLOBAL_COSTMAP_PERIOD:
                    costmap = build_global_costmap(grid.data, grid.resolution,
                                                   robot_radius=local_planning.ROBOT_RADIUS,
                                                   safety_clearance=GLOBAL_SAFETY_CLEARANCE_M)
                    safe_grid = costmap >= int(INSCRIBED_COST)
                    last_costmap_time = now

                # B: 인식 · 탐사
                image = perception.camera_image_to_bgr(camera.getImage(), camera.getWidth(), camera.getHeight())
                detections = [] if image is None else perception.detect_targets(image)
                perception_state, located = locate_targets(detections, lidar_points, pose, now, registry)
                grid.gcs_targets = registry.targets if registry is not None else ()   # GCS 사과 표시
                goal = selector.choose(grid, safe_grid, pose, now, allow_switch=False)

                # C: 전역 경로 (목표가 바뀌었거나 오래 막혔을 때만 다시 계획)
                if goal is None:
                    reset_path()
                else:
                    blocked = (path is not None and costmap is not None
                               and global_planning.path_is_blocked(grid, costmap, pose, path))
                    blocked_since = (blocked_since or now) if blocked else None
                    persistent = blocked_since is not None and now - blocked_since >= PATH_BLOCK_PERSISTENCE
                    urgent = goal_changed(goal, planned_goal) or persistent
                    if (path is None or urgent) and (urgent or now - last_plan_time >= REPLAN_RETRY_INTERVAL):
                        path = global_planning.plan(grid, costmap, pose, goal)
                        planned_goal, last_plan_time, blocked_since = (float(goal[0]), float(goal[1])), now, None

                # C: 지역 주행
                if goal is None:
                    linear, angular, state = 0.0, 0.0, 'WAIT_GOAL'
                elif path is None:
                    linear, angular, state = 0.0, 0.0, 'WAIT_PATH'
                else:
                    linear, angular, state = local_planning.command(pose, path, lidar_ranges)
                if state == 'ARRIVED' and goal is not None and selector.report_result(goal, 'ARRIVED', now):
                    reset_path()

                # Recovery: 갇히면 후진 → 회전. 반복 실패하면 탐사에 막힘을 알린다.
                override = recovery.update(now, pose, lidar_ranges, linear, angular, goal)
                if override is not None:
                    linear, angular = override
                    state = f'RECOVERY_{recovery.state}'
                if recovery.finished:
                    reset_path()
                    write_log(f't={now:.2f}s recovery finished count={recovery.count} '
                              f'gave_up={recovery.gave_up} goal={goal}')
                    if recovery.gave_up and goal is not None:
                        selector.report_result(goal, 'BLOCKED', now)

                # 모터
                left_speed, right_speed = local_planning.wheel_speeds(linear, angular)
                left_motor.setVelocity(max(-speed_limit, min(speed_limit, left_speed)))
                right_motor.setVelocity(max(-speed_limit, min(speed_limit, right_speed)))

                # 로그 · 지도 저장
                diagnostics.update(now, grid, safe_grid, pose, goal, write_log, selector=selector)
                if now >= next_log_time:
                    front = local_planning.nearest_front_obstacle(lidar_ranges)
                    write_log(f't={now:.2f}s state={state} mission=DISABLED '
                              f'front={"clear" if math.isinf(front) else f"{front:.2f}m"} '
                              f'pose=({pose.x:+.2f}, {pose.y:+.2f}, {math.degrees(pose.theta):+.1f}deg) '
                              f'goal={goal} detections={len(detections)} located={len(located)} '
                              f'targets={len(registry.targets) if registry else 0} perception={perception_state} '
                              f'slip={grid.slip_count}')
                    next_log_time += LOG_PERIOD
                if now >= next_map_time:
                    diagnostics.save_png(map_path, grid, pose)
                    write_log(f't={now:.2f}s map_saved={map_path}')
                    next_map_time += MAP_SAVE_PERIOD
        finally:
            if diagnostics.snapshot is not None:
                diagnostics.update(robot.getTime(), grid, safe_grid, pose, None, write_log, selector=selector)
                diagnostics.save_png(map_path, grid, pose)
            else:
                grid.save_png(map_path)
            write_log(f't={robot.getTime():.2f}s map_saved={map_path} final')


if __name__ == '__main__':
    main()
