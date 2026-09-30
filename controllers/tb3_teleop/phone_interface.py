"""Minimal phone camera/touch bridge for the Webots controller.

- No Flask dependency: uses Python's standard HTTP server.
- update_frame(image_bgr): publish the latest OpenCV BGR frame.
- get_click(): return the latest phone click once, then clear it.
- set_status(text): show a short robot/controller status on the phone.

The Webots Robot object must stay in the main controller thread.
This module only shares encoded images and click coordinates.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>TECH WEEK Robot Camera</title>
  <style>
    html, body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: system-ui, -apple-system, sans-serif;
    }
    .wrap {
      max-width: 900px;
      margin: 0 auto;
      padding: 12px;
    }
    h2 {
      margin: 4px 0 10px;
      font-size: 20px;
    }
    #cameraBox {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      touch-action: manipulation;
    }
    #camera {
      display: block;
      width: 100%;
      height: auto;
      user-select: none;
      -webkit-user-drag: none;
    }
    #marker {
      position: absolute;
      display: none;
      width: 20px;
      height: 20px;
      margin-left: -10px;
      margin-top: -10px;
      border: 3px solid #ff3b30;
      border-radius: 50%;
      box-sizing: border-box;
      pointer-events: none;
    }
    #status {
      margin-top: 10px;
      padding: 10px 12px;
      background: #222;
      border-radius: 8px;
      font-size: 15px;
      min-height: 22px;
    }
    #hint {
      margin-top: 8px;
      color: #aaa;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>TECH WEEK Robot Camera</h2>

    <div id="cameraBox">
      <img id="camera" src="/stream.mjpg" alt="robot camera">
      <div id="marker"></div>
    </div>

    <div id="status">연결 중...</div>
    <div id="hint">영상에서 원하는 위치를 누르면 카메라 픽셀 좌표가 로봇 controller로 전달됩니다.</div>
  </div>

<script>
const img = document.getElementById("camera");
const box = document.getElementById("cameraBox");
const marker = document.getElementById("marker");
const statusBox = document.getElementById("status");

async function sendPoint(clientX, clientY) {
  const rect = img.getBoundingClientRect();

  if (!img.naturalWidth || !img.naturalHeight ||
      clientX < rect.left || clientX > rect.right ||
      clientY < rect.top || clientY > rect.bottom) {
    return;
  }

  const nx = (clientX - rect.left) / rect.width;
  const ny = (clientY - rect.top) / rect.height;

  const x = Math.max(0, Math.min(img.naturalWidth - 1,
      Math.round(nx * img.naturalWidth)));
  const y = Math.max(0, Math.min(img.naturalHeight - 1,
      Math.round(ny * img.naturalHeight)));

  marker.style.display = "block";
  marker.style.left = ((clientX - rect.left) / rect.width * 100) + "%";
  marker.style.top = ((clientY - rect.top) / rect.height * 100) + "%";

  try {
    const r = await fetch("/click", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({x: x, y: y})
    });

    if (r.ok) {
      statusBox.textContent = `선택 좌표: (${x}, ${y})`;
    } else {
      statusBox.textContent = "좌표 전송 실패";
    }
  } catch (e) {
    statusBox.textContent = "서버 연결 실패";
  }
}

box.addEventListener("click", (e) => {
  sendPoint(e.clientX, e.clientY);
});

setInterval(async () => {
  try {
    const r = await fetch("/status", {cache: "no-store"});
    if (!r.ok) return;

    const data = await r.json();

    if (data.status && !statusBox.textContent.startsWith("선택 좌표:")) {
      statusBox.textContent = data.status;
    }
  } catch (e) {
    // The MJPEG stream may still be reconnecting. Keep the last text.
  }
}, 1000);
</script>
</body>
</html>
"""


class PhoneInterface:
    """Threaded HTTP interface for camera streaming and tap coordinates."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        jpeg_quality: int = 75,
        stream_fps: float = 10.0,
    ):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.stream_fps = float(stream_fps)

        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._click = None
        self._status = "카메라 연결됨"

        self._server = None
        self._thread = None

    def start(self):
        """Start the HTTP server in a daemon thread."""
        if self._server is not None:
            return

        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "TechWeekPhone/1.0"

            def log_message(self, fmt, *args):
                # Avoid filling the Webots console with HTTP access logs.
                return

            def _send_json(self, obj, status=200):
                payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    payload = _PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path == "/status":
                    with owner._lock:
                        status_text = owner._status
                        has_frame = owner._latest_jpeg is not None
                    self._send_json({
                        "status": status_text,
                        "has_frame": has_frame,
                    })
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()

                    last_seq = -1
                    min_period = 1.0 / max(owner.stream_fps, 1.0)

                    try:
                        while True:
                            started = time.monotonic()

                            with owner._frame_condition:
                                owner._frame_condition.wait_for(
                                    lambda: (
                                        owner._latest_jpeg is not None
                                        and owner._frame_seq != last_seq
                                    ),
                                    timeout=1.0,
                                )
                                jpeg = owner._latest_jpeg
                                seq = owner._frame_seq

                            if jpeg is None or seq == last_seq:
                                continue

                            last_seq = seq

                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()

                            elapsed = time.monotonic() - started
                            if elapsed < min_period:
                                time.sleep(min_period - elapsed)

                    except (
                        BrokenPipeError,
                        ConnectionResetError,
                        ConnectionAbortedError,
                        OSError,
                    ):
                        pass
                    return

                self.send_error(404)

            def do_POST(self):
                if self.path != "/click":
                    self.send_error(404)
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    data = json.loads(raw.decode("utf-8"))

                    x = int(data["x"])
                    y = int(data["y"])

                    if x < 0 or y < 0:
                        raise ValueError("negative coordinate")

                    with owner._lock:
                        # A newer tap replaces an older, unread tap.
                        owner._click = (x, y)

                    self._send_json({"ok": True, "x": x, "y": y})

                except Exception as exc:
                    self._send_json(
                        {"ok": False, "error": str(exc)},
                        status=400,
                    )

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="phone-http-server",
            daemon=True,
        )
        self._thread.start()

    def update_frame(self, image_bgr):
        """Encode and publish an OpenCV BGR image.

        Call this only from the Webots controller/main thread.
        """
        if image_bgr is None:
            return False

        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )

        if not ok:
            return False

        jpeg = encoded.tobytes()

        with self._frame_condition:
            self._latest_jpeg = jpeg
            self._frame_seq += 1
            self._frame_condition.notify_all()

        return True

    def get_click(self):
        """Return the newest (x, y) camera-pixel tap once, then clear it."""
        with self._lock:
            click = self._click
            self._click = None
        return click

    def set_status(self, text: str):
        """Update the short status shown on the phone webpage."""
        with self._lock:
            self._status = str(text)

    def stop(self):
        """Stop the HTTP server."""
        server = self._server
        self._server = None

        if server is not None:
            server.shutdown()
            server.server_close()

        thread = self._thread
        self._thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

