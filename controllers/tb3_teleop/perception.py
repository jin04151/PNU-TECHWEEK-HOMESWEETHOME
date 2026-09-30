"""B 담당: 빨간 사과 2개 구출을 위한 검출·위치 추정·목표 기록.

구출 완료는 통합 담당이 실제 판정 후 RESCUED 상태로 전달한다.
rescue_goal_reached는 구출 수 충족이며 시작점 복귀 완료를 뜻하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from localization import Pose


TARGET_LABEL = 'red_apple'
REQUIRED_RESCUES = 2


@dataclass
class Detection:
    """검출 결과. bbox는 왼쪽 위 원점의 (x_min, y_min, x_max, y_max) 픽셀 좌표.

    x는 오른쪽, y는 아래쪽으로 증가하며 최대 경계는 포함하지 않는다.
    confidence는 0~1의 후보 점수이며 확률이 아니다. 점수가 없으면 None을 쓴다.
    """

    label: str
    bbox: tuple[int, int, int, int]
    confidence: float | None = None
    # HSV 외곽선에서 얻은 영상 좌표이며 지도 좌표가 아니다.
    centroid: tuple[float, float] | None = None
    radius_px: float | None = None
    color_fraction: float | None = None


def camera_image_to_bgr(data: bytes | bytearray | memoryview | None,
                        width: int, height: int) -> np.ndarray | None:
    """Webots BGRA 버퍼를 (높이, 폭, 3)의 uint8 BGR 배열로 복사한다.

    왼쪽 위부터 행 순서로 저장된 영상이며 상하 반전은 하지 않는다.
    데이터가 없으면 None, 크기나 버퍼 길이가 잘못되면 ValueError를 발생시킨다.
    """
    if data is None:
        return None
    if any(isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer))
           or size <= 0 for size in (width, height)):
        raise ValueError('이미지 폭과 높이는 양의 정수여야 합니다.')
    pixels = np.frombuffer(data, dtype=np.uint8)
    if pixels.size != width * height * 4:
        raise ValueError('BGRA 데이터 길이는 폭 × 높이 × 4바이트여야 합니다.')
    # 공식 Camera 문서: getImage()는 픽셀당 B, G, R, A 순서의 4바이트.
    # https://cyberbotics.com/doc/reference/camera
    return pixels.reshape(height, width, 4)[:, :, :3].copy()


# OpenCV uint8 HSV: H=0..179. 실제 Webots 조명/거리에서 조정한다.
# 빨강은 Hue 양 끝에 걸치므로 두 범위를 합친다.
HSV_RANGES = {
    'red': (((0, 70, 40), (10, 255, 255)),
            ((170, 70, 40), (179, 255, 255))),
}


@dataclass(frozen=True)
class ColorConfig:
    """전체 영상의 빨간색 후보 필터. 면적 단위는 픽셀 제곱이다."""
    min_area_px: float = 20.0
    min_fraction: float = 0.35
    min_circularity: float = 0.30
    min_aspect_ratio: float = 0.40
    max_aspect_ratio: float = 2.50

    def __post_init__(self):
        values = (self.min_area_px, self.min_fraction, self.min_circularity,
                  self.min_aspect_ratio, self.max_aspect_ratio)
        if (not np.isfinite(values).all() or self.min_area_px <= 0
                or not 0 < self.min_fraction <= 1
                or not 0 <= self.min_circularity <= 1
                or not 0 < self.min_aspect_ratio <= self.max_aspect_ratio):
            raise ValueError('유효한 면적·색상 비율·원형도·가로세로 비율이 필요합니다.')


class AppleDetector:
    """OpenCV HSV 마스크와 외곽선으로 빨간 사과 후보를 검출한다.

    색상/형태 기반이므로 다른 빨간 물체도 후보가 될 수 있다.
    confidence는 색상 비율과 원형도의 평균 점수이며 학습 모델의 확률이 아니다.
    붙어 있는 빨간 영역은 하나의 외곽선으로 검출될 수 있다.
    """

    def __init__(self, *, color_config=None):
        self.color_config = color_config or ColorConfig()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """uint8 BGR 전체 영상에서 모든 유효 후보를 반환한다. 미검출은 []."""
        import cv2

        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or image.shape[0] == 0 or image.shape[1] == 0):
            raise ValueError('높이와 폭이 양수인 uint8 BGR (H,W,3) 영상이 필요합니다.')
        config = self.color_config
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        red_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in HSV_RANGES['red']:
            red_mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                                   np.array(upper, dtype=np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < config.min_area_px or perimeter <= 0:
                continue
            circularity = float(np.clip(4 * np.pi * area / perimeter ** 2, 0, 1))
            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / height
            # closing으로 채운 픽셀 대신 원래 HSV 마스크의 빨간색 픽셀 비율을 쓴다.
            fraction = cv2.countNonZero(red_mask[y:y + height, x:x + width]) / (width * height)
            if (circularity < config.min_circularity
                    or fraction < config.min_fraction
                    or not config.min_aspect_ratio <= aspect <= config.max_aspect_ratio):
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                continue
            centroid = (float(moments['m10'] / moments['m00']),
                        float(moments['m01'] / moments['m00']))
            _, radius = cv2.minEnclosingCircle(contour)
            score = float((fraction + circularity) / 2)
            detections.append(Detection(
                TARGET_LABEL, (x, y, x + width, y + height), score,
                centroid, float(radius), float(fraction)))
        # 프레임 내 순서를 고정할 뿐 ID는 아니다. 고유 ID는 TargetRegistry가 관리한다.
        detections.sort(key=lambda detection: detection.bbox[:2])
        return detections


_default_detector = None


def detect_targets(image: np.ndarray, *, detector=None) -> list[Detection]:
    """빨간 사과 후보만 반환한다. 두 개가 보여도 구출 완료로 간주하지 않는다.

    HSV 색상과 형태를 이용한 후보이며 사과 종류를 학습한 분류 결과는 아니다.
    반환 개수는 2개로 자르지 않는다. 고유 목표와 완료 수는 Registry가 관리한다.
    """
    global _default_detector
    if detector is None:
        if _default_detector is None:
            _default_detector = AppleDetector()
        detector = _default_detector
    return [detection for detection in detector.detect(image)
            if detection.label == TARGET_LABEL]


def detect(camera_image: bytes | None, width: int, height: int) -> Detection | None:
    """이전 단일 검출 호출용. 정확히 한 개일 때만 반환하며 다중 결과는 보류."""
    image = camera_image_to_bgr(camera_image, width, height)
    if image is None:
        return None
    detections = detect_targets(image)
    return detections[0] if len(detections) == 1 else None


@dataclass
class Calibration:
    """왜곡 보정된 핀홀 영상의 보정값. 행렬은 호출자가 확인해 제공한다.

    K는 3×3 픽셀 내부 행렬. T_camera_lidar는 LiDAR→광학 카메라,
    T_robot_lidar는 LiDAR→pose 기준 로봇의 4×4 강체 변환이다.
    광학 축은 x=영상 오른쪽, y=아래쪽, z=전방. 이동 단위는 m.
    로봇 축은 x=전방, y=좌측, z=위쪽이며 지도 변환은 평면 주행을 가정한다.
    """
    width: int
    height: int
    K: np.ndarray
    T_camera_lidar: np.ndarray
    T_robot_lidar: np.ndarray

    def __post_init__(self):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or v <= 0 for v in (self.width, self.height)):
            raise ValueError('영상 크기는 양의 정수여야 합니다.')
        self.K = np.array(self.K, dtype=float, copy=True)
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or not np.isclose(self.K[1, 0], 0)):
            raise ValueError('유효한 핀홀 내부 행렬이 필요합니다.')
        for name in ('T_camera_lidar', 'T_robot_lidar'):
            matrix = np.array(getattr(self, name), dtype=float, copy=True)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1])):
                raise ValueError('유한한 4×4 동차 변환이 필요합니다.')
            rotation = matrix[:3, :3]
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
                raise ValueError('회전 행렬은 직교하며 행렬식이 1이어야 합니다.')
            setattr(self, name, matrix)


@dataclass(frozen=True)
class Association:
    """검증 후 정할 대응 기준. 임의 운영 기본값은 두지 않는다."""
    min_points: int
    depth_gap_m: float
    max_depth_span_m: float
    max_time_delta_s: float

    def __post_init__(self):
        if isinstance(self.min_points, bool) or not isinstance(self.min_points, int) or self.min_points < 2:
            raise ValueError('최소 점 개수는 2 이상의 정수여야 합니다.')
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.depth_gap_m, self.max_depth_span_m, self.max_time_delta_s)):
            raise ValueError('대응 기준은 유한한 양수여야 합니다.')


@dataclass(frozen=True)
class ObservationTimes:
    """같은 시뮬레이션 시계의 초 단위 시각. lidar_s는 입력 점 순서와 같다."""
    camera_s: float
    pose_s: float
    lidar_s: tuple[float, ...]


def project_lidar_points(lidar_points, calibration: Calibration):
    """투영된 픽셀·카메라 깊이·로봇 점·원본 인덱스를 반환한다.

    입력은 Webots의 x/y/z 속성을 가진 점 목록 또는 (N,3) 배열이다.
    유효하지 않은 점·카메라 뒤·영상 밖 점은 제외한다. 왜곡 보정은 호출자 책임이다.
    """
    points = np.asarray([(p.x, p.y, p.z) if hasattr(p, 'x') else p
                         for p in lidar_points], dtype=float)
    if points.size == 0:
        points = np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('점군은 (N,3) 좌표여야 합니다.')
    indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    homogeneous = np.column_stack((points[indices], np.ones(len(indices))))
    camera = homogeneous @ calibration.T_camera_lidar.T
    robot = homogeneous @ calibration.T_robot_lidar.T
    front = camera[:, 2] > 0
    camera, robot, indices = camera[front], robot[front], indices[front]
    projected = camera[:, :3] @ calibration.K.T
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        pixels = projected[:, :2] / projected[:, 2:3]
    inside = (np.isfinite(pixels).all(axis=1) & np.isfinite(robot[:, :3]).all(axis=1)
              & (pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height))
    return pixels[inside], camera[inside, 2], robot[inside, :3], indices[inside]


def locate_target(detection: Detection, lidar_points, pose: Pose,
                  calibration: Calibration | None, *, association: Association | None = None,
                  timing: ObservationTimes | None = None, assume_static: bool = False
                  ) -> tuple[float, float] | None:
    """검출 상자에 대응하는 표면 대표점의 지도 (x,y)(m)를 추정한다.

    물체 중심이나 안전한 접근 위치를 보장하지 않는다. 대응 불가·여러 깊이군은 None.
    시각 자료가 없으면 None이며, 정지한 합성/정적 검증에서만 assume_static=True를 쓴다.
    시간차는 거부 조건일 뿐 이동 보정이 아니다. 회전형 LiDAR의 점별 시각을 전달해야 한다.
    """
    if calibration is None or association is None:
        return None
    if not np.isfinite([pose.x, pose.y, pose.theta]).all():
        return None
    box = np.asarray(detection.bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= calibration.width and 0 <= y0 < y1 <= calibration.height):
        return None
    lidar_points = list(lidar_points)
    pixels, depths, robot_points, indices = project_lidar_points(lidar_points, calibration)
    keep = ((pixels[:, 0] >= x0) & (pixels[:, 0] < x1)
            & (pixels[:, 1] >= y0) & (pixels[:, 1] < y1))
    if timing is None:
        if not assume_static:
            return None
    else:
        timestamps = np.asarray(timing.lidar_s, dtype=float)
        if timestamps.shape != (len(lidar_points),):
            return None
        if (not np.isfinite([timing.camera_s, timing.pose_s]).all()
                or abs(timing.camera_s-timing.pose_s) > association.max_time_delta_s):
            return None
        times = timestamps[indices]
        # 오래된 점을 일부 버린 뒤 배경만 선택하지 않도록 상자 안 시각이 하나라도 틀리면 거부.
        aligned = (np.isfinite(times)
                   & (np.abs(times-timing.camera_s) <= association.max_time_delta_s)
                   & (np.abs(times-timing.pose_s) <= association.max_time_delta_s))
        if np.any(keep & ~aligned):
            return None
    depths, robot_points = depths[keep], robot_points[keep]
    if len(depths) < association.min_points:
        return None
    order = np.argsort(depths)
    groups = np.split(order, np.flatnonzero(np.diff(depths[order]) > association.depth_gap_m)+1)
    # 작은 앞쪽 군도 실제 가림 물체일 수 있다. 여러 군 중 가까운/큰 군을 임의 선택하지 않는다.
    if len(groups) != 1 or np.ptp(depths) > association.max_depth_span_m:
        return None
    position = np.median(robot_points, axis=0)
    c, s = np.cos(pose.theta), np.sin(pose.theta)
    x = pose.x + c*position[0] - s*position[1]
    y = pose.y + s*position[0] + c*position[1]
    # TODO(B): 단일 배경면만 잡히는 경우를 구별하려면 목표물 마스크/크기 등 추가 근거가 필요하다.
    # TODO(B): 움직이는 관측은 자세 이력으로 점별 시각 보정 후 투영해야 한다.
    return float(x), float(y)


@dataclass(frozen=True)
class Target:
    """정적 목표물 기록. 위치는 pose와 같은 지도 좌표(m), 시각은 시뮬레이션 초."""
    target_id: int
    label: str
    position: tuple[float, float]
    last_seen: float
    observations: int = 1
    status: str = 'DISCOVERED'


class TargetRegistry:
    """프레임별 일대일 대응으로 목표물을 관리한다. 애매한 대응은 보류한다.

    match_distance_m은 호출자가 정할 위치 오차 기준이다. 실제 물체 ID를 보장하지 않는다.
    동일 좌표계의 정적 물체용이며 실행/좌표계가 바뀌면 새 객체를 만들어야 한다.
    """

    def __init__(self, match_distance_m: float, *, min_observations: int = 3):
        if not np.isfinite(match_distance_m) or match_distance_m <= 0:
            raise ValueError('중복 판단 거리는 유한한 양수여야 합니다.')
        if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 2:
            raise ValueError('확정에는 서로 다른 시각의 관측이 2회 이상 필요합니다.')
        self.match_distance_m = float(match_distance_m)
        self.min_observations = min_observations
        self._targets = {}
        self._next_id = 1
        self._last_frame_time = None
        self.last_reasons = ()

    @property
    def targets(self) -> tuple[Target, ...]:
        """확정 전 CANDIDATE를 포함한 기록. 접근 대상은 confirmed_targets 사용."""
        return tuple(self._targets.values())

    @property
    def confirmed_targets(self) -> tuple[Target, ...]:
        """반복 관측으로 확정된 빨간 사과 중 아직 구출하지 않은 목표."""
        return tuple(t for t in self.targets if t.label == TARGET_LABEL
                     and t.status not in ('CANDIDATE', 'RESCUED'))

    @property
    def rescued_count(self) -> int:
        """실제 구출 완료 통지를 받은 고유 목표 수. 발견/방문 횟수와 구분한다."""
        return sum(t.label == TARGET_LABEL and t.status == 'RESCUED' for t in self.targets)

    @property
    def rescue_goal_reached(self) -> bool:
        """빨간 사과 2개 구출 조건 충족. 귀환/임무 종료는 mission manager 담당."""
        return self.rescued_count >= REQUIRED_RESCUES

    def update(self, observations, now: float) -> list[int | None]:
        """한 프레임의 (Detection, 위치 또는 None) 목록을 받아 입력 순서의 ID를 반환한다.

        비목표 색·위치 추정 실패·대응 모호는 None이고 last_reasons에 이유를 남긴다.
        새 ID는 CANDIDATE이며 min_observations회 대응 후 DISCOVERED로 확정한다.
        프레임은 시간순으로 한 번씩 전달해야 한다. 잘못된 입력은 상태 변경 전에 거부한다.
        """
        if (not np.isfinite(now) or now < 0 or
                (self._last_frame_time is not None and now <= self._last_frame_time)):
            raise ValueError('프레임 시각은 음수가 아니며 이전 프레임보다 커야 합니다.')
        entries = []
        for detection, position in observations:
            if not isinstance(detection, Detection) or not isinstance(detection.label, str) or not detection.label:
                raise ValueError('종류 문자열이 있는 Detection이 필요합니다.')
            if position is not None:
                xy = np.asarray(position, dtype=float)
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    raise ValueError('물체 위치는 유한한 지도 좌표 (x,y)여야 합니다.')
                position = tuple(map(float, xy))
            entries.append((detection.label, position))

        # 갱신 전 기록으로 모든 후보를 계산한다. 입력 순서에 따른 탐욕적 병합을 피한다.
        candidates = []
        for label, position in entries:
            candidates.append([] if position is None or label != TARGET_LABEL else [t.target_id for t in self.targets
                if t.label == label and np.hypot(t.position[0]-position[0], t.position[1]-position[1])
                <= self.match_distance_m])
        claims = {target_id: sum(target_id in ids for ids in candidates) for target_id in self._targets}
        ids, reasons = [], []
        for index, ((label, position), possible) in enumerate(zip(entries, candidates)):
            if label != TARGET_LABEL:
                ids.append(None)
                reasons.append('구출 대상이 아님: 빨간 사과만 등록')
            elif position is None:
                ids.append(None)
                reasons.append('위치 추정 실패')
            elif len(possible) == 1 and claims[possible[0]] == 1:
                target = self._targets[possible[0]]
                observations_count = target.observations + 1
                status = target.status
                if status == 'CANDIDATE' and observations_count >= self.min_observations:
                    status = 'DISCOVERED'
                self._targets[target.target_id] = Target(target.target_id, target.label,
                    position, float(now), observations_count, status)
                ids.append(target.target_id)
                reasons.append('기존 목표 갱신')
            elif possible:
                ids.append(None)
                reasons.append('대응 모호: 여러 후보 또는 같은 목표에 여러 관측')
            else:
                # 가까운 새 관측은 두 물체인지 중복 검출인지 아직 구분할 수 없다.
                near = any(j != index and other is not None and other_label == label
                    and np.hypot(other[0]-position[0], other[1]-position[1]) <= self.match_distance_m
                    for j, (other_label, other) in enumerate(entries))
                if near:
                    ids.append(None)
                    reasons.append('대응 모호: 가까운 새 관측')
                    continue
                target_id = self._next_id
                self._next_id += 1
                self._targets[target_id] = Target(target_id, label, position, float(now),
                                                  status='CANDIDATE')
                ids.append(target_id)
                reasons.append('빨간 사과 후보 등록: 반복 관측 대기')
        self._last_frame_time = float(now)
        self.last_reasons = tuple(reasons)
        return ids

    def set_status(self, target_id: int, status: str) -> bool:
        """실제 접근/구출 판정은 호출자가 수행한다. RESCUED는 되돌리지 않는다.

        DISCOVERED→APPROACHING→RESCUED. VISITED는 구출 전 방문 상태로 호환한다.
        CANDIDATE 확정은 update()만 수행하며 접근 취소 시 DISCOVERED로 돌아간다.
        같은 상태의 반복 통지는 False, 없는 ID/잘못된 전이는 예외를 발생시킨다.
        """
        target = self._targets[target_id]
        if status == target.status:
            return False
        allowed = {'CANDIDATE': set(), 'DISCOVERED': {'APPROACHING'},
                   'APPROACHING': {'DISCOVERED', 'VISITED', 'RESCUED'},
                   'VISITED': {'RESCUED'}, 'RESCUED': set()}
        if status not in allowed[target.status]:
            raise ValueError('허용되지 않은 목표 상태 전이입니다.')
        self._targets[target_id] = Target(target.target_id, target.label, target.position,
                                         target.last_seen, target.observations, status)
        return True
