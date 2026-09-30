"""TurtleBot3 SLAM 실시간 GCS.

controllers/tb3_teleop/mapping.py 가 grid.update() 때마다 UDP(127.0.0.1:5600)로
지도·자세·LiDAR 점을 보낸다. 이 창은 그걸 받아 그리고, Webots 실행도 여기서 한다.

    python gcs.py            # 창만 띄운다
    python gcs.py --run      # 창을 띄우고 바로 Webots 실행

필요한 것: PyQt6, numpy.
"""

import argparse
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtNetwork import QHostAddress, QUdpSocket
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QPlainTextEdit, QPushButton, QSplitter,
                             QVBoxLayout, QWidget)


HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get('AMR_GCS_PORT', '5600'))
DEFAULT_WORLD = 'worlds/apartment.wbt'      # gcs.py 위치 기준


def find_webots():
    """WEBOTS_HOME → PATH 의 webots → OS 기본 설치 위치 순서로 찾는다."""
    import shutil
    home = os.environ.get('WEBOTS_HOME')
    candidates = []
    if home:
        candidates += [Path(home) / 'msys64/mingw64/bin/webots.exe', Path(home) / 'webots']
    found = shutil.which('webots')
    if found:
        candidates.append(Path(found))
    if sys.platform == 'win32':
        candidates.append(Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Webots/msys64/mingw64/bin/webots.exe')
        candidates.append(Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Webots/msys64/mingw64/bin/webots.exe')
    else:
        candidates += [Path('/usr/local/webots/webots'), Path('/snap/bin/webots'), Path.home() / 'webots/webots']
    return next((path for path in candidates if path.exists()), None)


def resolve(path_text):
    """상대 경로는 gcs.py 가 있는 폴더 기준으로 푼다."""
    path = Path(path_text)
    return path if path.is_absolute() else HERE / path


WEBOTS = find_webots()
TRAIL_LENGTH = 20000
LOG_DIR = HERE / 'gcs_logs'
NO_DATA_WARNING_S = 90.0      # apartment 월드는 로딩만 50초 안팎 걸린다
ERROR_MARKERS = ('Traceback', 'Error', 'error:', 'Exception', 'crashed', 'exited with status')

MAP_UNKNOWN = (58, 60, 66)
MAP_FREE = (214, 216, 220)
MAP_OCCUPIED = (18, 18, 20)
ROBOT_COLOR = QColor('#4aa3ff')
SCAN_COLOR = QColor(255, 80, 80, 200)
# 사과 상태별 색. CANDIDATE: 확인 중, DISCOVERED: 발견(확정), RESCUED: 구조 완료.
TARGET_COLORS = {'CANDIDATE': QColor('#ffb340'), 'DISCOVERED': QColor('#ff3b30'), 'RESCUED': QColor('#4cd964')}
TARGET_NAMES = {'CANDIDATE': '확인 중', 'DISCOVERED': '발견', 'RESCUED': '구조'}


class MapView(QWidget):
    """월드 좌표(m)를 그린다. +x 오른쪽, +y 위쪽."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)
        self.image = None
        self.meta = None
        self.trail = []
        self.pose = None
        self.robot_radius = 0.10
        self.scan = []
        self.targets = []
        self.layers = {'map': True, 'scan': True, 'grid': True, 'trail': True, 'targets': True}
        self.center = [0.0, 0.0]
        self.scale = 60.0
        self.follow = False
        self.auto_fit = True
        self._drag = None
        self._mouse = None

    def reset(self):
        self.image = None
        self.meta = None
        self.trail.clear()
        self.pose = None
        self.scan = []
        self.targets = []
        self.auto_fit = True
        self.update()

    def set_map(self, meta, data):
        rows, cols = meta['rows'], meta['cols']
        grid = np.frombuffer(data, dtype=np.int8).reshape(rows, cols)
        rgb = np.empty((rows, cols, 3), dtype=np.uint8)
        rgb[:] = MAP_UNKNOWN
        rgb[grid == 0] = MAP_FREE
        rgb[grid == 100] = MAP_OCCUPIED
        self._rgb = np.ascontiguousarray(np.flipud(rgb))
        self.image = QImage(self._rgb.data, cols, rows, 3 * cols, QImage.Format.Format_RGB888)
        self.meta = meta

    def set_state(self, state):
        self.pose = state['pose']
        self.robot_radius = state.get('robot_radius', self.robot_radius)
        self.scan = state.get('scan', [])
        self.targets = state.get('targets', [])
        self.trail.append((self.pose[0], self.pose[1]))
        if len(self.trail) > TRAIL_LENGTH:
            del self.trail[:len(self.trail) - TRAIL_LENGTH]
        if self.auto_fit and self.pose is not None:
            self.center = [self.pose[0], self.pose[1]]
            self.auto_fit = False
        if self.follow:
            self.center = [self.pose[0], self.pose[1]]
        self.update()

    def fit(self):
        if self.meta is None:
            return
        # 지도에서 이미 알려진(빈 곳·장애물) 영역에 맞춘다.
        data = np.frombuffer(self._rgb.tobytes(), np.uint8).reshape(self.meta['rows'], self.meta['cols'], 3)
        known = np.flipud((data != MAP_UNKNOWN).any(axis=2))
        rows, cols = np.nonzero(known)
        res = self.meta['resolution']
        if len(rows) == 0:
            return
        xs = (cols - self.meta['origin_col']) * res
        ys = (rows - self.meta['origin_row']) * res
        self.center = [(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2]
        span = max(xs.max() - xs.min(), ys.max() - ys.min(), 2.0) + 1.0
        self.scale = 0.95 * min(self.width(), self.height()) / span
        self.update()

    def to_screen(self, x, y):
        return QPointF(self.width() / 2 + (x - self.center[0]) * self.scale,
                       self.height() / 2 - (y - self.center[1]) * self.scale)

    def to_world(self, point):
        return (self.center[0] + (point.x() - self.width() / 2) / self.scale,
                self.center[1] - (point.y() - self.height() / 2) / self.scale)

    def wheelEvent(self, event):
        before = self.to_world(event.position())
        self.scale = max(5.0, min(3000.0, self.scale * (1.2 if event.angleDelta().y() > 0 else 1 / 1.2)))
        after = self.to_world(event.position())
        self.center[0] += before[0] - after[0]
        self.center[1] += before[1] - after[1]
        self.update()

    def mousePressEvent(self, event):
        self._drag = (event.position(), list(self.center))

    def mouseMoveEvent(self, event):
        self._mouse = self.to_world(event.position())
        if self._drag is not None:
            start, center = self._drag
            delta = event.position() - start
            self.center = [center[0] - delta.x() / self.scale, center[1] + delta.y() / self.scale]
        self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#1c1d21'))

        if self.layers['map'] and self.image is not None:
            meta = self.meta
            res = meta['resolution']
            top_left = self.to_screen((-meta['origin_col'] - 0.5) * res,
                                      (meta['rows'] - 1 - meta['origin_row'] + 0.5) * res)
            painter.drawImage(QRectF(top_left.x(), top_left.y(),
                                     meta['cols'] * res * self.scale,
                                     meta['rows'] * res * self.scale), self.image)

        if self.layers['grid']:
            painter.setPen(QPen(QColor(255, 255, 255, 28), 1))
            x0, y1 = self.to_world(QPointF(0, 0))
            x1, y0 = self.to_world(QPointF(self.width(), self.height()))
            if x1 - x0 < 400:
                for gx in range(math.floor(x0), math.ceil(x1) + 1):
                    painter.drawLine(self.to_screen(gx, y0), self.to_screen(gx, y1))
                for gy in range(math.floor(y0), math.ceil(y1) + 1):
                    painter.drawLine(self.to_screen(x0, gy), self.to_screen(x1, gy))

        if self.layers['targets'] and self.targets:
            painter.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
            for target_id, label, x, y, status in self.targets:
                color = TARGET_COLORS.get(status, QColor('#ff3b30'))
                centre = self.to_screen(x, y)
                radius = max(7.0, 0.12 * self.scale)
                painter.setPen(QPen(QColor('#ffffff'), 2))
                painter.setBrush(color)
                painter.drawEllipse(centre, radius, radius)
                painter.setPen(QPen(QColor('#2e7d32'), 3))           # 사과 꼭지
                painter.drawLine(centre + QPointF(0, -radius), centre + QPointF(radius * 0.4, -radius * 1.5))
                painter.setPen(QColor('#ffffff'))
                painter.drawText(centre + QPointF(radius + 4, 4),
                                 f'#{target_id} {TARGET_NAMES.get(status, status)}')

        if self.layers['scan'] and self.scan:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(SCAN_COLOR)
            dot = max(1.5, min(3.0, 0.02 * self.scale))
            for x, y in self.scan:
                painter.drawEllipse(self.to_screen(x, y), dot, dot)

        if self.layers['trail'] and len(self.trail) > 1:
            path = QPainterPath(self.to_screen(*self.trail[0]))
            for x, y in self.trail[1:]:
                path.lineTo(self.to_screen(x, y))
            painter.setPen(QPen(ROBOT_COLOR, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        if self.pose is not None:
            x, y, theta = self.pose
            centre = self.to_screen(x, y)
            radius = self.robot_radius * self.scale
            painter.setPen(QPen(ROBOT_COLOR, 2))
            painter.setBrush(QColor(74, 163, 255, 90))
            painter.drawEllipse(centre, radius, radius)
            nose = self.to_screen(x + self.robot_radius * math.cos(theta),
                                  y + self.robot_radius * math.sin(theta))
            painter.setPen(QPen(QColor('#ffffff'), 2))
            painter.drawLine(centre, nose)

        painter.setFont(QFont('Consolas', 9))
        painter.setPen(QColor('#bdbdbd'))
        if self._mouse is not None:
            painter.drawText(12, self.height() - 10,
                             f'x={self._mouse[0]:+.2f} m  y={self._mouse[1]:+.2f} m   '
                             '(휠: 확대, 드래그: 이동, 더블클릭: 지도에 맞춤)')
        bar = self.scale
        right = self.width() - 16
        painter.setPen(QPen(QColor('#e6e6e6'), 2))
        painter.drawLine(QPointF(right - bar, self.height() - 16), QPointF(right, self.height() - 16))
        painter.drawText(QPointF(right - bar / 2 - 10, self.height() - 22), '1 m')
        painter.drawText(12, 18, f'로봇: 반지름 {self.robot_radius * 100:.0f} cm 원')
        painter.end()


class GCS(QMainWindow):

    def __init__(self, world):
        super().__init__()
        self.setWindowTitle('TurtleBot3 SLAM GCS')
        self.resize(1400, 900)
        self.view = MapView()
        self.process = None
        self.run_id = None
        self.last_packet = None
        self.packets = 0
        self.ignored = set()
        self.announced = set()
        self.started_at = None
        self.warned_no_data = False
        self.issue = None
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / f'gcs_{time.strftime("%Y%m%d_%H%M%S")}.log'
        self.log_file = self.log_path.open('a', encoding='utf-8', buffering=1)
        self._build(world)
        self.log(f'[GCS] 로그 파일: {self.log_path}')

        self.socket = QUdpSocket(self)
        if not self.socket.bind(QHostAddress.SpecialAddress.LocalHost, PORT):
            self._set_issue(f'UDP {PORT} 포트를 다른 프로그램이 쓰는 중 (GCS 가 이미 켜져 있나요?)')
        self.socket.readyRead.connect(self._receive)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(500)
        self.log(f'[GCS] UDP 127.0.0.1:{PORT} 수신 대기.')

    def _build(self, world):
        side = QWidget()
        layout = QVBoxLayout(side)

        run_box = QGroupBox('Webots 실행')
        form = QFormLayout(run_box)
        row = QHBoxLayout()
        self.world_edit = QLineEdit(str(world))
        browse = QPushButton('…')
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse)
        row.addWidget(self.world_edit)
        row.addWidget(browse)
        form.addRow('월드', row)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['실시간 (Webots 화면 표시)', '고속 (Webots 화면 없음)'])
        form.addRow('모드', self.mode_combo)
        buttons = QHBoxLayout()
        self.start_button = QPushButton('▶ 실행')
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton('■ 정지')
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        form.addRow(buttons)
        layout.addWidget(run_box)

        status_box = QGroupBox('상태')
        grid = QGridLayout(status_box)
        self.labels = {}
        for index, (key, name) in enumerate((('link', '연결'), ('time', '경과'), ('pose', '위치'),
                                             ('heading', '방향'), ('match', '스캔 정합'),
                                             ('map', '지도 칸'), ('targets', '사과'), ('issue', '문제'))):
            grid.addWidget(QLabel(name), index, 0)
            label = QLabel('-')
            label.setFont(QFont('Consolas', 10))
            grid.addWidget(label, index, 1)
            self.labels[key] = label
        layout.addWidget(status_box)

        layer_box = QGroupBox('표시')
        layer_layout = QGridLayout(layer_box)
        for index, (key, name) in enumerate((('map', '지도'), ('scan', 'LiDAR 점'),
                                             ('grid', '1 m 격자'), ('trail', '궤적'),
                                             ('targets', '사과'))):
            check = QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(lambda on, k=key: self._layer(k, on))
            layer_layout.addWidget(check, 0, index)
        follow = QCheckBox('로봇 따라가기')
        follow.toggled.connect(lambda on: setattr(self.view, 'follow', on))
        layer_layout.addWidget(follow, 1, 0, 1, 2)
        clear = QPushButton('궤적 지우기')
        clear.clicked.connect(lambda: (self.view.trail.clear(), self.view.update()))
        layer_layout.addWidget(clear, 1, 2, 1, 2)
        layout.addWidget(layer_box)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont('Consolas', 9))
        self.log_view.setMaximumBlockCount(3000)
        log_box = QGroupBox('로그')
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_box, 1)

        splitter = QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(side)
        splitter.setSizes([950, 450])
        self.setCentralWidget(splitter)

        save = QAction('지도 화면 저장', self)
        save.setShortcut('Ctrl+S')
        save.triggered.connect(self._save)
        self.menuBar().addMenu('파일').addAction(save)

    def _layer(self, key, on):
        self.view.layers[key] = on
        self.view.update()

    def log(self, text):
        self.log_view.appendPlainText(text)
        try:
            self.log_file.write(f'{time.strftime("%H:%M:%S")} {text}\n')
        except (OSError, ValueError):
            pass

    def _set_issue(self, text):
        self.issue = text
        self.labels['issue'].setText(f'<span style="color:#ff5f57">{text}</span>')
        self.log(f'[GCS][문제] {text}')

    def _receive(self):
        while self.socket.hasPendingDatagrams():
            payload = bytes(self.socket.receiveDatagram().data())
            if len(payload) < 8 or payload[:4] != b'AMR2':
                continue
            length = struct.unpack('<I', payload[4:8])[0]
            try:
                state = json.loads(payload[8:8 + length].decode('utf-8'))
            except ValueError:
                continue
            self._handle(state, payload[8 + length:])

    def _handle(self, state, blob):
        run = state.get('run')
        stale = self.last_packet is None or time.monotonic() - self.last_packet > 2.0
        if run != self.run_id:
            if not stale:
                if run not in self.ignored:
                    self.ignored.add(run)
                    self.log(f'[GCS] 다른 실행(ID {run})의 데이터는 무시합니다. Webots 가 두 개 켜져 있나요?')
                return
            self.run_id = run
            self.announced = set()
            self.view.reset()
            self.log(f'[GCS] 실행 ID {run} 수신 시작.')
        self.last_packet = time.monotonic()
        self.packets += 1

        if blob and 'map' in state:
            try:
                first = self.view.meta is None
                self.view.set_map(state['map'], zlib.decompress(blob))
                if first:
                    QTimer.singleShot(0, self.view.fit)
            except (zlib.error, ValueError):
                pass
        self.view.set_state(state)

        x, y, theta = state['pose']
        self.labels['time'].setText(f'{state.get("wall", 0):.1f} s')
        self.labels['pose'].setText(f'x {x:+.2f} m   y {y:+.2f} m')
        self.labels['heading'].setText(f'{math.degrees(theta):+.1f}°')
        done, failed = state.get('match', 0), state.get('failed', 0)
        rate = f'{100 * done / (done + failed):.0f}%' if done + failed else '-'
        self.labels['match'].setText(f'{done}/{done + failed} 성공 ({rate}), 응답 {state.get("response", 0):.2f}')
        targets = state.get('targets', [])
        counts = {name: sum(1 for t in targets if t[4] == key) for key, name in TARGET_NAMES.items()}
        new = [t for t in targets if t[4] != 'CANDIDATE' and t[0] not in self.announced]
        for target_id, label, x, y, status in new:
            self.announced.add(target_id)
            self.log(f'[GCS] 사과 #{target_id} {TARGET_NAMES.get(status, status)}: ({x:+.2f}, {y:+.2f}) m')
        self.labels['targets'].setText(' · '.join(f'{name} {n}' for name, n in counts.items()) if targets
                                       else '아직 없음')
        free, occupied, unknown = state.get('counts', (0, 0, 0))
        self.labels['map'].setText(f'빈 곳 {free} · 장애물 {occupied} · 모름 {unknown}')

    def _tick(self):
        if self.last_packet is None:
            self.labels['link'].setText('● 연결 없음')
            if (self.process is not None and self.started_at is not None and not self.warned_no_data
                    and time.monotonic() - self.started_at > NO_DATA_WARNING_S):
                self.warned_no_data = True
                self._set_issue(f'실행 {NO_DATA_WARNING_S:.0f}초가 지나도 데이터 없음')
                self.log('[GCS] 원인 후보: (0) 월드 로딩이 아직 안 끝남(apartment 는 50초 안팎), (1) 로봇 컨트롤러가 시작하자마자 죽음 — 위 로그의 Traceback 확인, '
                         '(2) 컨트롤러가 mapping.grid.update() 를 부르지 않음, '
                         '(3) AMR_GCS=0 으로 꺼져 있음, (4) 포트 불일치')
            return
        age = time.monotonic() - self.last_packet
        self.labels['link'].setText('<span style="color:#4cd964">● 수신 중</span>' if age < 1.5
                                    else f'<span style="color:#ff5f57">● 끊김 ({age:.0f}s 전)</span>')

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, '월드 선택', str(resolve(self.world_edit.text())),
                                              'Webots world (*.wbt)')
        if path:
            self.world_edit.setText(path)

    def start(self):
        if self.process is not None:
            self.log('[GCS] 이미 실행 중입니다.')
            return
        world = resolve(self.world_edit.text())
        if not world.exists():
            self.log(f'[GCS] 월드 파일이 없습니다: {world}')
            return
        if WEBOTS is None:
            self.log('[GCS] Webots 를 찾을 수 없습니다. 환경변수 WEBOTS_HOME 을 설치 폴더로 지정하세요.')
            return
        env = QProcessEnvironment.systemEnvironment()
        for name in ('QT_QPA_PLATFORM', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.remove(name)          # Webots 도 Qt 앱이라 이 창의 설정을 물려받으면 안 뜬다
        env.insert('PYTHONIOENCODING', 'utf-8')
        env.insert('AMR_GCS_PORT', str(PORT))
        args = ['--stdout', '--stderr']
        args += (['--mode=fast', '--no-rendering', '--minimize'] if self.mode_combo.currentIndex() == 1
                 else ['--mode=realtime'])
        args.append(str(world))
        self.run_id = None
        self.ignored = set()
        self.view.reset()
        self.process = QProcess(self)
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.start(str(WEBOTS), args)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.started_at = time.monotonic()
        self.warned_no_data = False
        self.issue = None
        self.labels['issue'].setText('-')
        self.log(f'[GCS] 실행: {world} ({self.mode_combo.currentText()}), 포트 {PORT}')

    def _read(self):
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', 'replace')
        for line in text.splitlines():
            line = line.rstrip()
            if not line.strip() or 'minimal requirements' in line.lower():
                continue
            self.log(line)
            if any(marker in line for marker in ERROR_MARKERS) and 'WARNING' not in line:
                if self.issue is None or not self.issue.startswith('컨트롤러 오류'):
                    self._set_issue(f'컨트롤러 오류: {line.strip()[:120]}')
                elif 'Error' in line and ':' in line:
                    self._set_issue(f'컨트롤러 오류: {line.strip()[:120]}')

    def _finished(self, code, status):
        self.log(f'[GCS] Webots 종료 (코드 {code}), 받은 패킷 {self.packets}개.')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process = None

    def stop(self):
        if self.process is None:
            return
        pid = self.process.processId()
        if pid and sys.platform == 'win32':
            # webots.exe 는 실행기라 시뮬레이터와 컨트롤러가 자식으로 남는다.
            QProcess.execute('taskkill', ['/F', '/T', '/PID', str(pid)])
        self.process.kill()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, '지도 화면 저장', 'gcs_map.png', 'PNG (*.png)')
        if path:
            self.view.grab().save(path)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


def dark(app):
    app.setStyle('Fusion')
    palette = QPalette()
    base, window, text = QColor('#232428'), QColor('#2b2d31'), QColor('#e6e6e6')
    for role, color in ((QPalette.ColorRole.Window, window), (QPalette.ColorRole.WindowText, text),
                        (QPalette.ColorRole.Base, base), (QPalette.ColorRole.AlternateBase, window),
                        (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, window),
                        (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.Highlight, QColor('#4aa3ff'))):
        palette.setColor(role, color)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor('#777'))
    app.setPalette(palette)


def main():
    parser = argparse.ArgumentParser(description='TurtleBot3 SLAM GCS')
    parser.add_argument('--run', action='store_true', help='창을 띄우자마자 Webots 실행')
    parser.add_argument('--world', default=str(DEFAULT_WORLD))
    parser.add_argument('--fast', action='store_true', help='Webots 화면 없이 고속')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    dark(app)
    window = GCS(args.world)
    window.mode_combo.setCurrentIndex(1 if args.fast else 0)
    window.show()
    if args.run:
        QTimer.singleShot(500, window.start)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
