"""안전한 Frontier를 BFS로 찾고 관측량·거리·회전·방문 점수로 평가한다.

참고: SeanReg/nav2_wavefront_frontier_exploration의
nav2_wfd/wavefront_frontier.py: isFrontierPoint(), getFrontier().
경계 판정과 BFS 그룹화 개념을 참고해 현재 지도 규칙으로 작성했다.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from mapping import FREE, UNKNOWN, OCCUPIED


def _neighbors(cell, shape):
    """지도 내부의 상하좌우 셀을 반환한다."""
    row, col = cell
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield nr, nc


def _frontier_mask(data):
    """Unknown과 변을 공유하는 Free 셀만 경계로 표시한다."""
    unknown = data == UNKNOWN
    adjacent = np.zeros(data.shape, dtype=bool)
    adjacent[1:] |= unknown[:-1]
    adjacent[:-1] |= unknown[1:]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    return (data == FREE) & adjacent


def _groups(mask):
    """4방향으로 연결된 경계 셀들을 BFS로 묶는다."""
    remaining = mask.copy()
    for row, col in np.argwhere(mask):
        start = (int(row), int(col))
        if not remaining[start]:
            continue
        remaining[start] = False
        queue = deque([start])
        group = []
        while queue:
            cell = queue.popleft()
            group.append(cell)
            for neighbor in _neighbors(cell, mask.shape):
                if remaining[neighbor]:
                    remaining[neighbor] = False
                    queue.append(neighbor)
        yield group


def _search(grid, safe_grid, pose):
    """최단 이동 칸 수와 해당 BFS 경로의 첫 진행 방향(rad)을 구한다."""
    blocked = np.asarray(safe_grid)
    if blocked.shape != grid.data.shape or blocked.dtype != np.bool_:
        raise ValueError('safe_grid는 지도와 같은 크기의 bool 배열이어야 합니다.')
    distance = np.full(grid.data.shape, -1, dtype=np.int32)
    heading = np.full(grid.data.shape, np.nan)
    if not all(math.isfinite(v) for v in (pose.x, pose.y)):
        return distance, heading
    start = grid.world_to_grid(pose.x, pose.y)
    if start is None or grid.data[start] != FREE:
        return distance, heading
    walkable = (grid.data == FREE) & ~blocked
    # FREE인 현재 위치만 inflation 차단 예외로 둔다. 원본 safe_grid는 유지한다.
    # 주변 셀은 기존 안전 조건을 만족해야 BFS가 확장된다.
    walkable[start] = True
    distance[start] = 0
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        neighbors = list(_neighbors(cell, grid.data.shape))
        if cell == start:
            # 최단 경로가 여럿이면 첫 회전이 작은 방향부터 탐색한다.
            neighbors.sort(key=lambda n: _angle_difference(
                math.atan2(n[0] - cell[0], n[1] - cell[1]), pose.theta))
        for neighbor in neighbors:
            if walkable[neighbor] and distance[neighbor] < 0:
                distance[neighbor] = distance[cell] + 1
                heading[neighbor] = (math.atan2(neighbor[0] - cell[0], neighbor[1] - cell[1])
                                     if cell == start else heading[cell])
                queue.append(neighbor)
    return distance, heading


def _angle_difference(a, b):
    """두 방향의 최소 각도 차이(0~π)를 반환한다."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _distances(grid, safe_grid, pose):
    """기존 진단에서 사용하는 BFS 거리 배열이다. 미도달은 -1이다."""
    return _search(grid, safe_grid, pose)[0]


def goal_is_valid(grid, safe_grid, pose, goal, frontier=True):
    """기존 목표가 안전하게 도달 가능하고 아직 경계인지 확인한다."""
    cell = grid.world_to_grid(*goal)
    if cell is None:
        return False
    if frontier and not _frontier_mask(grid.data)[cell]:
        return False
    return bool(_distances(grid, safe_grid, pose)[cell] >= 0)


def choose_nearest_goal(grid, safe_grid, pose) -> tuple[float, float] | None:
    """최단 BFS 거리의 경계 셀 중심을 지도 좌표(m)로 반환한다.

    그룹 평균 대신 실제 셀을 선택한다. 동률은 행·열 순서로 정한다.
    현재 도착 범위(반 셀) 안의 후보는 제외한다.
    None은 현재 후보 없음이며 전체 임무 완료를 뜻하지 않는다.
    """
    distances = _distances(grid, safe_grid, pose)
    representatives = []
    for group in _groups(_frontier_mask(grid.data)):
        candidates = []
        for row, col in group:
            if distances[row, col] < 0:
                continue
            x, y = grid.grid_to_world(row, col)
            if math.hypot(x - pose.x, y - pose.y) <= grid.resolution / 2:
                continue
            candidates.append((int(distances[row, col]), row, col))
        if candidates:
            representatives.append(min(candidates))
    if not representatives:
        return None
    _, row, col = min(representatives)
    return grid.grid_to_world(row, col)


@dataclass(frozen=True)
class ScoringConfig:
    """실험용 초기값. 거리·반경은 m, 시간은 시뮬레이션 초 단위다."""

    observation_radius_m: float = 1.5
    distance_scale_m: float = 3.0
    gain_weight: float = 3.0
    distance_weight: float = 1.0
    turn_weight: float = 0.3
    revisit_weight: float = 1.0
    revisit_radius_m: float = 0.5
    revisit_decay_s: float = 30.0
    switch_margin: float = 0.15
    retry_delay_s: float = 5.0
    # 구역 유지: 고른 경계 주변을 한 구역으로 보고, 구역에 경계가 남아 있는 동안
    # 칸 단위로 목표를 바꾸지 않는다(매초 A* 재계획·맴돌기 방지).
    commit_radius_m: float = 1.0
    arrive_radius_m: float = 0.35
    blocked_grace_s: float = 2.0
    # 진척 없음 감지: 구역을 쫓는 동안 알려진 면적이 progress_window_s 동안
    # min_progress_m2 이상 늘지 않으면 그 구역을 blacklist_duration_s 동안 뺀다.
    progress_window_s: float = 25.0
    min_progress_m2: float = 0.5
    blacklist_radius_m: float = 1.0
    blacklist_duration_s: float = 120.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}은 유한한 0 이상의 값이어야 합니다.')
        for name in ('observation_radius_m', 'distance_scale_m', 'revisit_radius_m',
                     'revisit_decay_s', 'retry_delay_s', 'commit_radius_m',
                     'progress_window_s', 'blacklist_radius_m', 'blacklist_duration_s'):
            if getattr(self, name) == 0:
                raise ValueError(f'{name}은 0보다 커야 합니다.')


@dataclass(frozen=True)
class Candidate:
    """평가한 실제 셀과 점수 근거. 평균 좌표는 사용하지 않는다."""

    cell: tuple[int, int]
    group: int
    goal: tuple[float, float]
    gain_m2: float
    distance_m: float
    turn_rad: float
    revisit: float
    score: float


def evaluate_candidates(grid, safe_grid, pose, config=None, visits=(), now=0.0):
    """안전한 후보 모두의 점수를 계산한다. visits는 ((x,y),시각) 목록이다."""
    config = config or ScoringConfig()
    distances, headings = _search(grid, safe_grid, pose)
    if not math.isfinite(pose.theta):
        return []
    radius = math.ceil(config.observation_radius_m / grid.resolution)
    offsets = np.arange(-radius, radius + 1)
    disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2) * grid.resolution ** 2 <= config.observation_radius_m ** 2
    disk_cells = int(disk.sum())
    candidates = []
    for group_id, group in enumerate(_groups(_frontier_mask(grid.data))):
        for row, col in group:
            if distances[row, col] < 0:
                continue
            goal = grid.grid_to_world(row, col)
            if math.dist(goal, (pose.x, pose.y)) <= grid.resolution / 2:
                continue
            r0, r1 = max(0, row-radius), min(grid.rows, row+radius+1)
            c0, c1 = max(0, col-radius), min(grid.cols, col+radius+1)
            window = disk[r0-row+radius:r1-row+radius, c0-col+radius:c1-col+radius]
            # 주변 Unknown 면적 근사치다. 벽 차폐·센서 시야각은 아직 반영하지 않는다.
            unknown_count = int(np.count_nonzero((grid.data[r0:r1, c0:c1] == UNKNOWN) & window))
            distance_m = float(distances[row, col]) * grid.resolution
            first_heading = headings[row, col]
            if not math.isfinite(first_heading):
                first_heading = math.atan2(goal[1]-pose.y, goal[0]-pose.x)
            turn = _angle_difference(first_heading, pose.theta)
            penalty = max((
                max(0.0, 1 - math.dist(goal, point) / config.revisit_radius_m)
                * max(0.0, 1 - (now - time) / config.revisit_decay_s)
                for point, time in visits if 0 <= now-time < config.revisit_decay_s
            ), default=0.0)
            score = (config.gain_weight * unknown_count / disk_cells
                     - config.distance_weight * distance_m / config.distance_scale_m
                     - config.turn_weight * turn / math.pi
                     - config.revisit_weight * penalty)
            candidates.append(Candidate((row, col), group_id, goal,
                                        unknown_count * grid.resolution ** 2,
                                        distance_m, turn, penalty, score))
    return candidates


def _rank(candidate):
    """동점이면 가까운 셀, 행·열 순으로 고른다."""
    return -candidate.score, candidate.distance_m, candidate.cell


def choose_goal(grid, safe_grid, pose) -> tuple[float, float] | None:
    """관측량·거리·첫 회전 점수로 목표를 고른다. 상태는 저장하지 않는다."""
    candidates = evaluate_candidates(grid, safe_grid, pose)
    return min(candidates, key=_rank).goal if candidates else None


class FrontierSelector:
    """방문 이력과 목표 유지를 관리한다. main에서 실행별로 한 번 생성한다."""

    def __init__(self, config=None):
        self.config = config or ScoringConfig()
        self.goal = None
        self.visits = []
        self.candidates = []
        self.reason = '초기 상태'
        self._last_time = None
        self._pending_reason = None
        self._retry_after = {}
        self.region = None               # 지금 쫓는 구역의 기준점 (x, y)
        self.blacklist = []              # [((x, y), 해제 시각)]
        self.reached = []                # 도착했지만 아직 경계인 목표 [((x, y), 해제 시각)]
        self._progress_area = 0.0
        self._progress_since = 0.0
        self._empty_since = None

    def _time(self, now):
        if not math.isfinite(now):
            raise ValueError('시각은 유한한 시뮬레이션 초여야 합니다.')
        if self._last_time is not None and now < self._last_time:
            self.__init__(self.config)
        self._last_time = now

    def report_result(self, goal, status, now):
        """현재 탐색 목표의 실제 결과만 받는다. 반복·지난 목표 이벤트는 무시한다."""
        if status not in ('ARRIVED', 'BLOCKED', 'PLAN_FAILED'):
            raise ValueError('ARRIVED, BLOCKED, PLAN_FAILED만 전달할 수 있습니다.')
        self._time(now)
        if goal is None or self.goal is None or tuple(goal) != self.goal:
            return False
        if status == 'ARRIVED':
            self.visits.append((self.goal, now))
            self._pending_reason = '도착 확인 후 재선택'
        else:
            # 같은 실패 목표를 다음 주기에 곧바로 반복 선택하지 않는다.
            self._retry_after[self.goal] = now + self.config.retry_delay_s
            self._pending_reason = '차단 후 재선택' if status == 'BLOCKED' else '계획 실패 후 재선택'
        self.goal = None
        return True

    def _blacklisted(self, point):
        return (any(math.dist(point, center) <= self.config.blacklist_radius_m
                    for center, _ in self.blacklist)
                or any(math.dist(point, center) <= self.config.arrive_radius_m
                       for center, _ in self.reached))

    def _commit(self, candidate, known_m2, now):
        """새 구역을 쫓기 시작한다. 진척 감시도 새로 시작한다."""
        self.goal = candidate.goal
        self.region = candidate.goal
        self._progress_area = known_m2
        self._progress_since = now

    def choose(self, grid, safe_grid, pose, now, *, allow_switch=True):
        """최신 지도에서 목표를 고른다.

        한 구역(commit_radius_m)을 고르면 그 구역에 경계가 남아 있는 동안 목표를
        칸 단위로 바꾸지 않는다. 구역을 쫓는 동안 알려진 면적이 늘지 않으면
        (progress_window_s 동안 min_progress_m2 미만) 그 구역을 잠시 뺀다.
        """
        config = self.config
        self._time(now)
        self.visits = [(point, time) for point, time in self.visits
                       if now-time < config.revisit_decay_s]
        self._retry_after = {point: time for point, time in self._retry_after.items() if now < time}
        self.blacklist = [(center, until) for center, until in self.blacklist if now < until]
        self.reached = [(center, until) for center, until in self.reached if now < until]
        self.candidates = evaluate_candidates(grid, safe_grid, pose, config, self.visits, now)
        known_m2 = float(np.count_nonzero(grid.data != UNKNOWN)) * grid.resolution ** 2
        position = (pose.x, pose.y)

        # A. 진척 없음 감지: 같은 구역을 쫓는데 지도가 늘지 않으면 그 구역을 뺀다.
        if self.region is not None:
            if known_m2 - self._progress_area >= config.min_progress_m2:
                self._progress_area, self._progress_since = known_m2, now
            elif now - self._progress_since >= config.progress_window_s:
                self.blacklist.append((self.region, now + config.blacklist_duration_s))
                self._pending_reason = (f'진척 없음 {config.progress_window_s:.0f}s: 구역 '
                                        f'({self.region[0]:.1f},{self.region[1]:.1f}) '
                                        f'{config.blacklist_duration_s:.0f}s 제외')
                self.region = None
                self.goal = None

        eligible = [c for c in self.candidates
                    if c.goal not in self._retry_after and not self._blacklisted(c.goal)]

        # 시작 셀이 잠깐 차단되는 등으로 후보 계산이 안 되면 현재 목표를 잠시 유지한다.
        if not self.candidates and self.goal is not None:
            if self._empty_since is None:
                self._empty_since = now
            if now - self._empty_since < config.blocked_grace_s:
                self.reason = '후보 계산 불가(시작 셀 차단 등): 현재 목표 잠시 유지'
                return self.goal
        else:
            self._empty_since = None

        if (self.goal is not None and math.dist(self.goal, position) <= grid.resolution / 2
                and goal_is_valid(grid, safe_grid, pose, self.goal)):
            self.reason = '도착 범위: 통합 담당의 도착 확인 대기'
            return self.goal

        previous = self.goal
        if not eligible:
            self.goal = None
            self.region = None
            self.reason = self._pending_reason or (
                '실패·진척없음 구역 대기' if self.candidates else '유효한 후보 없음')
            self._pending_reason = None
            return None

        # D. 구역 유지: 구역에 경계가 남아 있으면 그 안에서만 움직인다.
        if self.region is not None:
            local = [c for c in eligible if math.dist(c.goal, self.region) <= config.commit_radius_m]
            if local:
                arrived = self.goal is not None and math.dist(self.goal, position) <= config.arrive_radius_m
                still_frontier = any(c.goal == self.goal for c in eligible)
                if (self.goal is not None and not arrived
                        and (still_frontier or goal_is_valid(grid, safe_grid, pose, self.goal,
                                                             frontier=False))):
                    self.reason = '구역 유지: 현재 목표 유지'
                    self._pending_reason = None
                    return self.goal
                if arrived:
                    # 가 봤는데도 아직 경계인 칸은 한동안 다시 고르지 않는다(칸 사이 핑퐁 방지).
                    self.reached.append((self.goal, now + config.blacklist_duration_s))
                    local = [c for c in local if not self._blacklisted(c.goal)]
                ahead = [c for c in local if math.dist(c.goal, position) > config.arrive_radius_m]
                if not ahead:
                    self.region = None
                    return self.choose_new_region(grid, pose, eligible, known_m2, now, previous)
                self.goal = min(ahead, key=_rank).goal
                self.region = self.goal
                self.reason = self._pending_reason or '구역 유지: 같은 구역의 다음 경계'
                self._pending_reason = None
                return self.goal

        # 구역이 없거나 다 풀렸으면 전체에서 새 구역을 고른다.
        current = next((c for c in eligible if c.goal == self.goal), None)
        best = min(eligible, key=_rank)
        if current is not None and (not allow_switch
                                    or best.score <= current.score + self.config.switch_margin):
            self._commit(current, known_m2, now)
            self.reason = '유효한 현재 목표 유지'
            self._pending_reason = None
            return self.goal
        return self.choose_new_region(grid, pose, eligible, known_m2, now, previous)

    def choose_new_region(self, grid, pose, eligible, known_m2, now, previous):
        """제외 목록을 뺀 후보 중 최고 점수로 새 구역을 시작한다."""
        eligible = [c for c in eligible if not self._blacklisted(c.goal)]
        if not eligible:
            self.goal = None
            self.region = None
            self.reason = self._pending_reason or '실패·진척없음 구역 대기'
            self._pending_reason = None
            return None
        self._commit(min(eligible, key=_rank), known_m2, now)
        self.reason = self._pending_reason or ('첫 목표 선택' if previous is None
                                               else '구역 소진 후 새 구역 선택')
        self._pending_reason = None
        return self.goal


class Diagnostics:
    """통합 담당이 매 주기 갱신하는 탐색 진단. 주행 판단은 변경하지 않는다."""

    def __init__(self):
        self.distance_m = 0.0
        self._previous_position = None
        self._previous_time = None
        self._previous_goal = None
        self._next_log_time = 0.0
        self._pending_events = []
        self.snapshot = None

    def update(self, now, grid, safe_grid, pose, goal, write_log, selector=None):
        """시뮬레이션 시각·실제 선택 목표를 받아 기존 로그에 1초마다 기록한다."""
        if not math.isfinite(now):
            raise ValueError('시뮬레이션 시각은 유한한 값이어야 합니다.')
        if self._previous_time is not None and now < self._previous_time:
            self.__init__()
        position = (float(pose.x), float(pose.y))
        valid_position = all(math.isfinite(v) for v in position)
        if valid_position and self._previous_position is not None:
            self.distance_m += math.dist(position, self._previous_position)
        self._previous_position = position if valid_position else None
        self._previous_time = now

        frontier = _frontier_mask(grid.data)
        distances = _distances(grid, safe_grid, pose)
        reachable = frontier & (distances >= 0)
        candidates = reachable.copy()
        for row, col in np.argwhere(reachable):
            if math.dist(grid.grid_to_world(row, col), position) <= grid.resolution / 2:
                candidates[row, col] = False
        start = grid.world_to_grid(*position) if valid_position else None
        if not valid_position:
            reason = '로봇 위치가 유효하지 않음'
        elif start is None:
            reason = '시작 셀이 지도 밖'
        elif grid.data[start] == UNKNOWN:
            reason = '시작 셀이 Unknown'
        elif grid.data[start] != FREE:
            reason = '시작 셀이 Free가 아님'
        elif not frontier.any():
            reason = 'Frontier 없음'
        elif not reachable.any():
            reason = '도달 가능한 후보 없음'
        elif not candidates.any():
            reason = '도달 가능한 경계가 모두 현재 도착 범위 안'
        else:
            reason = '안전한 Frontier 후보 있음'

        current_goal = tuple(goal) if goal is not None else None
        if current_goal != self._previous_goal:
            if current_goal is None:
                change_reason = reason if not candidates.any() else '호출자가 목표를 지정하지 않음'
            elif self._previous_goal is None:
                change_reason = '호출자가 첫 목표를 선택함'
            else:
                change_reason = '호출자가 목표를 변경함'
            # 실제 도착이나 계획 실패 여부는 목표 변화만으로 추정하지 않는다.
            self._pending_events.append(
                f't={now:.2f}s goal_change={self._previous_goal}->{current_goal} reason={change_reason}')
        self._previous_goal = current_goal
        # 선택 객체가 없으면 방문 이력 없이 기본 점수를 계산한다.
        scores = (selector.candidates if selector is not None
                  else evaluate_candidates(grid, safe_grid, pose))
        selection_reason = selector.reason if selector is not None else '상태 없는 점수 선택'
        self.snapshot = {
            'frontier': frontier, 'reachable': reachable,
            'candidate_count': int(candidates.sum()),
            'group_count': sum(1 for _ in _groups(frontier)),
            'known_area_m2': float(np.count_nonzero(
                (grid.data == FREE) | (grid.data == OCCUPIED)) * grid.resolution ** 2),
            'goal': current_goal, 'reason': reason,
            'scores': scores, 'selection_reason': selection_reason,
        }
        if now >= self._next_log_time:
            for event in self._pending_events:
                write_log(event)
            self._pending_events.clear()
            self._next_log_time = math.floor(now) + 1.0
            info = self.snapshot
            goal_text = 'None' if current_goal is None else f'({current_goal[0]:.3f},{current_goal[1]:.3f})'
            write_log(
                f't={now:.2f}s exploration known_area_m2={info["known_area_m2"]:.3f} '
                f'frontier_groups={info["group_count"]} reachable_candidates={info["candidate_count"]} '
                f'goal={goal_text} odom_distance_m={self.distance_m:.3f} '
                f'reason={reason} selection_reason={selection_reason} '
                f'goal_reached=N/A planning_failures=N/A')
            # 전체 후보는 snapshot에 보관하고 로그에는 상위 5개와 현재 목표를 남긴다.
            shown = sorted(scores, key=_rank)[:5]
            selected = next((c for c in scores if c.goal == current_goal), None)
            if selected is not None and selected not in shown:
                shown.append(selected)
            for candidate in shown:
                write_log(
                    f't={now:.2f}s frontier_score cell={candidate.cell} '
                    f'goal={candidate.goal} gain_m2={candidate.gain_m2:.3f} '
                    f'distance_m={candidate.distance_m:.3f} turn_rad={candidate.turn_rad:.3f} '
                    f'revisit={candidate.revisit:.3f} score={candidate.score:.3f} '
                    f'selected={candidate.goal == current_goal}')
        return self.snapshot

    def save_png(self, path, grid, pose):
        """최근 진단을 같은 PNG에 덮어쓴다. +x는 오른쪽, +y는 위쪽이다."""
        from PIL import Image, ImageDraw

        if self.snapshot is None:
            raise RuntimeError('PNG 저장 전에 update()를 호출해야 합니다.')
        pixels = np.full((*grid.data.shape, 3), 128, dtype=np.uint8)
        pixels[grid.data == FREE] = (255, 255, 255)
        pixels[grid.data == OCCUPIED] = (0, 0, 0)
        pixels[self.snapshot['frontier']] = (0, 0, 255)
        pixels[self.snapshot['reachable']] = (0, 255, 0)
        image = Image.fromarray(np.flipud(pixels).copy())
        draw = ImageDraw.Draw(image)
        if all(math.isfinite(v) for v in (pose.x, pose.y, pose.theta)):
            cell = grid.world_to_grid(pose.x, pose.y)
            if cell is not None:
                row, col = cell
                px, py = col, grid.rows - 1 - row
                # 화면 y축은 아래로 증가하므로 sin 부호를 뒤집는다.
                end = (px + 6 * math.cos(pose.theta), py - 6 * math.sin(pose.theta))
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=(255, 165, 0))
                draw.line((px, py, *end), fill=(255, 165, 0), width=1)
                for offset in (-.6, .6):
                    angle = pose.theta + offset
                    wing = (end[0] - 3 * math.cos(angle), end[1] + 3 * math.sin(angle))
                    draw.line((*end, *wing), fill=(255, 165, 0), width=1)
        goal = self.snapshot['goal']
        if goal is not None:
            cell = grid.world_to_grid(*goal)
            if cell is not None:
                row, col = cell
                draw.point((col, grid.rows - 1 - row), fill=(255, 0, 0))
        image.save(path)
