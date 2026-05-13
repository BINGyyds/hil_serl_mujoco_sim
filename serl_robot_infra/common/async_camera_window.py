from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Optional, Sequence

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RenderSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    time: float
    act: Optional[np.ndarray] = None
    mocap_pos: Optional[np.ndarray] = None
    mocap_quat: Optional[np.ndarray] = None


class _AsyncMultiCameraRenderer:
    """Render the latest MuJoCo state from a background thread.

    The worker always drops stale states and only renders the newest snapshot.
    This keeps UI rendering from back-pressuring the control loop.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        camera_names: Sequence[str],
        width: int,
        height: int,
        fps: int,
        cv2_module,
    ) -> None:
        self.model = model
        self.camera_names = tuple(camera_names)
        self.width = int(width)
        self.height = int(height)
        self.frame_interval = 1.0 / float(fps)
        self.cv2 = cv2_module

        self._condition = threading.Condition()
        self._stop_requested = False
        self._latest_snapshot: Optional[_RenderSnapshot] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._last_render_wall_time = 0.0
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="async-camera-renderer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit_snapshot(self, data: mujoco.MjData) -> None:
        snapshot = _RenderSnapshot(
            qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(data.qvel, dtype=np.float64).copy(),
            ctrl=np.asarray(data.ctrl, dtype=np.float64).copy(),
            time=float(data.time),
            act=np.asarray(data.act, dtype=np.float64).copy()
            if getattr(data, "act", None) is not None and data.act.size
            else None,
            mocap_pos=np.asarray(data.mocap_pos, dtype=np.float64).copy()
            if getattr(data, "mocap_pos", None) is not None and data.mocap_pos.size
            else None,
            mocap_quat=np.asarray(data.mocap_quat, dtype=np.float64).copy()
            if getattr(data, "mocap_quat", None) is not None and data.mocap_quat.size
            else None,
        )
        with self._condition:
            self._latest_snapshot = snapshot
            self._condition.notify_all()

    def get_latest_frame(self) -> tuple[Optional[np.ndarray], int, Optional[str]]:
        with self._condition:
            return self._latest_frame, self._latest_frame_id, self._last_error

    def _run(self) -> None:
        local_data = mujoco.MjData(self.model)
        renderer: Optional[mujoco.Renderer] = None
        next_render_time = time.perf_counter()

        try:
            renderer = mujoco.Renderer(self.model, self.height, self.width)
            while True:
                with self._condition:
                    while not self._stop_requested and self._latest_snapshot is None:
                        self._condition.wait(timeout=0.1)
                    if self._stop_requested:
                        return
                    snapshot = self._latest_snapshot

                now = time.perf_counter()
                if now < next_render_time:
                    time.sleep(min(next_render_time - now, 0.005))
                    continue

                if snapshot is None:
                    next_render_time = time.perf_counter() + self.frame_interval
                    continue

                self._apply_snapshot(local_data, snapshot)
                frame = self._render_panel(renderer, local_data)
                with self._condition:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._last_render_wall_time = time.perf_counter()
                    self._last_error = None

                next_render_time = max(
                    next_render_time + self.frame_interval,
                    time.perf_counter(),
                )
        except Exception as exc:
            logger.exception("异步相机渲染线程失败。")
            with self._condition:
                self._last_error = str(exc)
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    logger.exception("关闭异步相机渲染器失败。")

    def _apply_snapshot(self, local_data: mujoco.MjData, snapshot: _RenderSnapshot) -> None:
        np.copyto(local_data.qpos, snapshot.qpos)
        np.copyto(local_data.qvel, snapshot.qvel)
        np.copyto(local_data.ctrl, snapshot.ctrl)
        if snapshot.act is not None and local_data.act.size == snapshot.act.size:
            np.copyto(local_data.act, snapshot.act)
        if snapshot.mocap_pos is not None and local_data.mocap_pos.size == snapshot.mocap_pos.size:
            np.copyto(local_data.mocap_pos, snapshot.mocap_pos)
        if snapshot.mocap_quat is not None and local_data.mocap_quat.size == snapshot.mocap_quat.size:
            np.copyto(local_data.mocap_quat, snapshot.mocap_quat)
        local_data.time = snapshot.time
        mujoco.mj_forward(self.model, local_data)

    def _render_panel(
        self,
        renderer: mujoco.Renderer,
        local_data: mujoco.MjData,
    ) -> np.ndarray:
        display_frames = []
        for camera_name in self.camera_names:
            renderer.update_scene(local_data, camera=camera_name)
            frame = renderer.render()
            bgr_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
            self.cv2.putText(
                bgr_frame,
                camera_name,
                (8, 22),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
            display_frames.append(bgr_frame)

        if not display_frames:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return np.hstack(display_frames)


class AsyncCameraWindow:
    """OpenCV window backed by an asynchronous MuJoCo camera renderer."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_names: Sequence[str],
        width: int,
        height: int,
        fps: int,
        window_name: str,
    ) -> None:
        if fps <= 0:
            raise ValueError("camera fps must be positive.")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("未安装 OpenCV(cv2)，无法打开相机窗口。") from exc

        self.model = model
        self.data = data
        self.camera_names = tuple(camera_names)
        self.width = int(width)
        self.height = int(height)
        self.window_name = window_name
        self.cv2 = cv2
        self.window_open = True
        self._last_displayed_frame_id = -1
        self._last_reported_error: Optional[str] = None

        missing = [
            name
            for name in self.camera_names
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0
        ]
        if missing:
            raise ValueError(f"MuJoCo 模型缺少相机: {missing}")

        self._renderer = _AsyncMultiCameraRenderer(
            model=model,
            camera_names=self.camera_names,
            width=self.width,
            height=self.height,
            fps=int(fps),
            cv2_module=cv2,
        )

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width * len(self.camera_names), height)
        self._renderer.start()
        self.submit_latest_state()

    def submit_latest_state(self) -> None:
        self._renderer.submit_snapshot(self.data)

    def render(self) -> None:
        if not self.window_open:
            return

        self.submit_latest_state()
        frame, frame_id, error_text = self._renderer.get_latest_frame()
        if error_text and error_text != self._last_reported_error:
            logger.error("相机窗口后台渲染失败: %s", error_text)
            self._last_reported_error = error_text

        if frame is not None and frame_id != self._last_displayed_frame_id:
            self.cv2.imshow(self.window_name, frame)
            self._last_displayed_frame_id = frame_id

        key = self.cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.close()

    def reset_timing(self) -> None:
        """保持与旧调用点兼容。"""
        self.submit_latest_state()

    def close(self) -> None:
        if self.window_open:
            self._renderer.stop()
            try:
                self.cv2.destroyWindow(self.window_name)
            except Exception:
                logger.exception("关闭相机窗口失败。")
            self.window_open = False
