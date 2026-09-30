"""벽·장애물에 붙어 못 움직일 때 빠져나오는 복구 동작 (Nav2 의 BackUp → Spin 방식).

매 loop 에 update() 를 부르면, 평소에는 None 을 돌려주고 로컬 플래너 명령을 그대로
쓰게 한다. 갇혔다고 판단하면 (linear, angular) 를 돌려주며 아래 순서로 움직인다.

  1. BACKUP : 뒤가 비어 있으면 천천히 후진해 벽에서 떨어진다.
  2. SPIN   : LiDAR 로 가장 트인 방향을 찾아 그쪽으로 돈다.
  3. 끝     : finished=True. 호출자는 경로를 다시 계획한다.
             같은 목표에서 여러 번 실패하면 gave_up=True 로 알려 목표를 바꾸게 한다.

LiDAR 거리 배열은 local_costmap 과 같은 규칙이다: angle = π - 2π·i/개수 (정면 = 가운데).
"""

import math


# 갇힘 판단
STUCK_DISTANCE_M = 0.05
STUCK_NEAR_TIME_S = 6.0          # 앞이 가까울 때: 이 시간 동안 못 움직이면 갇힘
STUCK_ANY_TIME_S = 10.0          # 그 외: 이 시간 동안 못 움직이면 갇힘
NEAR_FRONT_M = 0.30

# 후진
BACKUP_SPEED = 0.08              # m/s
BACKUP_DISTANCE_M = 0.20
BACKUP_TIMEOUT_S = 3.0
REAR_CLEARANCE_M = 0.20          # 뒤쪽 ±35° 안의 가장 가까운 장애물이 이보다 멀어야 후진

# 회전
SPIN_SPEED = 1.0                 # rad/s
SPIN_TIMEOUT_S = 4.0
SPIN_TOLERANCE_RAD = 0.15
SECTOR_WIDTH_RAD = math.radians(30)

# 반복 실패
COOLDOWN_S = 3.0
GIVE_UP_COUNT = 3
GIVE_UP_WINDOW_S = 60.0


def _angle(index, count):
    return math.pi - 2.0 * math.pi * index / count


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sector_clearance(ranges, center, half_width):
    """center(rad, 로봇 기준) ± half_width 안의 가장 가까운 거리."""
    count = len(ranges)
    nearest = math.inf
    for index, value in enumerate(ranges):
        if math.isfinite(value) and value > 0.0 and abs(_wrap(_angle(index, count) - center)) <= half_width:
            nearest = min(nearest, value)
    return nearest


def most_open_direction(ranges):
    """가장 트인 방향(rad, 로봇 기준). 비슷하면 덜 도는 쪽을 고른다."""
    best, best_score = 0.0, -math.inf
    for step in range(-18, 18):
        center = math.radians(step * 10)
        clearance = min(sector_clearance(ranges, center, SECTOR_WIDTH_RAD / 2), 3.5)
        score = clearance - 0.1 * abs(center)
        if score > best_score:
            best, best_score = center, score
    return best


class Recovery:

    def __init__(self):
        self.state = 'IDLE'
        self.finished = False
        self.gave_up = False
        self.count = 0
        self._history = []           # [(t, x, y)] 명령을 내리는 동안의 위치
        self._started = 0.0
        self._start_xy = (0.0, 0.0)
        self._spin_target = 0.0
        self._cooldown_until = -math.inf
        self._attempts = []          # [(t, goal)]

    def update(self, now, pose, ranges, linear, angular, goal):
        """복구 중이면 (linear, angular) 를, 아니면 None 을 돌려준다."""
        self.finished = False
        self.gave_up = False
        ranges = list(ranges or [])

        if self.state == 'BACKUP':
            moved = math.hypot(pose.x - self._start_xy[0], pose.y - self._start_xy[1])
            rear = sector_clearance(ranges, math.pi, math.radians(35))
            if moved >= BACKUP_DISTANCE_M or now - self._started >= BACKUP_TIMEOUT_S or rear < REAR_CLEARANCE_M:
                self._begin_spin(now, pose, ranges)
            else:
                return -BACKUP_SPEED, 0.0

        if self.state == 'SPIN':
            error = _wrap(self._spin_target - pose.theta)
            if abs(error) <= SPIN_TOLERANCE_RAD or now - self._started >= SPIN_TIMEOUT_S:
                self._finish(now, goal)
                return 0.0, 0.0
            return 0.0, math.copysign(SPIN_SPEED, error)

        if goal is None or now < self._cooldown_until or not (abs(linear) > 0.01 or abs(angular) > 0.05):
            self._history.clear()
            return None

        self._history.append((now, pose.x, pose.y))
        self._history = [h for h in self._history if now - h[0] <= STUCK_ANY_TIME_S]
        if self._stuck(now, pose, ranges):
            self.count += 1
            self._attempts.append((now, tuple(goal)))
            rear = sector_clearance(ranges, math.pi, math.radians(35))
            if rear >= REAR_CLEARANCE_M:
                self.state = 'BACKUP'
                self._started = now
                self._start_xy = (pose.x, pose.y)
                return -BACKUP_SPEED, 0.0
            self._begin_spin(now, pose, ranges)
            return 0.0, 0.0
        return None

    def _stuck(self, now, pose, ranges):
        front = sector_clearance(ranges, 0.0, math.radians(20))
        window = STUCK_NEAR_TIME_S if front < NEAR_FRONT_M else STUCK_ANY_TIME_S
        old = [h for h in self._history if now - h[0] >= window]
        if not old:
            return False
        _, x, y = old[-1]
        return math.hypot(pose.x - x, pose.y - y) < STUCK_DISTANCE_M

    def _begin_spin(self, now, pose, ranges):
        self.state = 'SPIN'
        self._started = now
        self._spin_target = _wrap(pose.theta + most_open_direction(ranges))

    def _finish(self, now, goal):
        self.state = 'IDLE'
        self.finished = True
        self._history.clear()
        self._cooldown_until = now + COOLDOWN_S
        if goal is not None:
            recent = [t for t, g in self._attempts
                      if now - t <= GIVE_UP_WINDOW_S and math.dist(g, goal) < 0.3]
            self.gave_up = len(recent) >= GIVE_UP_COUNT
