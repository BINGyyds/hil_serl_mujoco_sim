import threading
import time
import numpy as np
from typing import Tuple


class ZeroSpaceMouseExpert:
    """用于无设备调试的零输入 SpaceMouse 替身。"""

    def get_action(self) -> Tuple[np.ndarray, list]:
        return np.zeros(6, dtype=np.float64), [0, 0]

    def close(self) -> None:
        pass


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        try:
            from . import pyspacemouse
        except ModuleNotFoundError as exc:
            if exc.name == "easyhid":
                raise RuntimeError(
                    "SpaceMouse 依赖缺失: 未安装 `easyhid`。"
                    " 请在当前环境中执行 `pip install easyhid`，并确认系统已安装 HID 相关库后重试。"
                ) from exc
            raise

        self.pyspacemouse = pyspacemouse
        try:
            device = pyspacemouse.open()
        except Exception as exc:
            message = str(exc)
            if "HID API is probably not installed" in message or "hid_enumerate" in message:
                raise RuntimeError(
                    "SpaceMouse HID 后端不可用: `easyhid` 已安装，但系统 `hidapi` 动态库未正确安装"
                    " 或当前环境无法找到它。请先修复本机 hidapi，再连接 SpaceMouse。"
                ) from exc
            if "Failed to open device" in message:
                raise RuntimeError(
                    "SpaceMouse 已被枚举到，但打开设备失败。"
                    " 在 Apple Silicon macOS 上，请优先使用 `bglopez/python-easyhid` 这个 patched easyhid fork；"
                    " 如果安装了 3Dconnexion 官方驱动，请先退出 `3DconnexionHelper` 后重试；"
                    " 若 ARM 原生环境仍失败，再尝试 Rosetta 2 / x86_64 Python。"
                ) from exc
            if "No device connected/supported!" in message:
                raise RuntimeError("未检测到可用的 SpaceMouse 设备。请连接设备后重试。") from exc
            raise RuntimeError(f"SpaceMouse 初始化失败: {message}") from exc

        if device is None:
            raise RuntimeError("未检测到可用的 SpaceMouse 设备。")

        # Use a lightweight background thread instead of multiprocessing.
        # macOS defaults to the "spawn" start method, which requires pickling
        # the target object. This class stores imported modules and HID handles,
        # so a process target is fragile here while a thread is sufficient.
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.latest_action = [0.0] * 6
        self.latest_buttons = [0, 0, 0, 0]

        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    def _read_spacemouse(self):
        while not self._stop_event.is_set():
            state = self.pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if state is None:
                time.sleep(0.001)
                continue

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            with self._lock:
                self.latest_action = action
                self.latest_buttons = buttons

            time.sleep(0.001)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self._lock:
            action = list(self.latest_action)
            buttons = list(self.latest_buttons)
        return np.array(action), buttons
    
    def close(self):
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.pyspacemouse.close()
