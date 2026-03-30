# async_writer.py

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local OpenCV install
    cv2 = None

logger = logging.getLogger(__name__)

FRAME_SENTINEL = None

class AsyncVideoWriter:
    """
    一个在独立的后台线程中将视频帧异步写入文件的类。

    该类通过将耗时的文件I/O操作转移到专用线程，来避免阻塞主仿真循环，
    是长时间视频录制的理想选择。它的内存占用非常低且恒定。
    """
    def __init__(self, video_path: Path, base_fps: int, frame_size: Tuple[int, int], 
                 speed_multiplier: float = 1.0):
        """
        初始化异步视频写入器。

        Args:
            video_path (Path): 视频文件的输出路径。
            base_fps (int): 视频的采样帧率（仿真中每秒渲染的帧数）。
            frame_size (Tuple[int, int]): 视频帧的尺寸 (宽度, 高度)。
            speed_multiplier (float): 视频播放的倍速。
        """
        if cv2 is None:
            raise RuntimeError("未安装 OpenCV(cv2)，无法启用视频写入。")

        # **倍速功能实现**: 视频的最终播放帧率是 采样率 * 倍速
        output_fps = base_fps * speed_multiplier
        logger.info(f"视频 '{video_path.name}' 配置: [采样率: {base_fps} FPS], "
                    f"[倍速: {speed_multiplier}x], [最终输出帧率: {output_fps:.2f} FPS]")

        # 创建一个有界队列，防止内存无限增长
        self.queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=base_fps * 10) # 增加到10秒缓冲区
        
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_writer: Any = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            output_fps, # <-- 使用计算出的新帧率
            frame_size
        )
        # 守护线程确保主程序退出时它也会退出
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.is_running = False

    def start(self):
        """启动后台写入线程。"""
        if not self.is_running:
            self.is_running = True
            self.thread.start()
            logger.info(f"视频写入线程已为 '{self.video_writer.getBackendName()}' 启动。")

    def add_frame(self, frame: np.ndarray):
        """
        从主线程向队列中添加一帧图像（非阻塞）。
        """
        if self.is_running:
            try:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.queue.put_nowait(bgr_frame)
            except queue.Full:
                logger.warning("视频写入队列已满，为保持仿真流畅性，丢弃一帧。")

    def stop(self):
        """
        平滑地停止写入器，等待所有缓存帧被写入文件。
        """
        if self.is_running:
            logger.info(f"正在停止视频写入器 for '{self.video_writer.getBackendName()}'...")
            self.is_running = False
            self.queue.put(FRAME_SENTINEL)
            self.thread.join()
            logger.info("视频写入器已成功停止。")

    def _write_loop(self):
        """后台线程的主循环，负责从队列取帧并写入文件。"""
        while True:
            frame = self.queue.get()
            if frame is FRAME_SENTINEL:
                break
            self.video_writer.write(frame)
        
        self.video_writer.release()
        logger.info("视频文件已成功保存并关闭。")

class AsyncImageWriter:
    QueueItem = Optional[Tuple[Path, np.ndarray]]

    def __init__(self):
        if cv2 is None:
            raise RuntimeError("未安装 OpenCV(cv2)，无法启用图片写入。")

        self.queue: queue.Queue[AsyncImageWriter.QueueItem] = queue.Queue(maxsize=20)
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.is_running = False

    def start(self):
        self.is_running = True
        self.thread.start()

    def save_image(self, path: Path, frame: np.ndarray):
        if self.is_running:
            try:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.queue.put_nowait((path, bgr_frame))
            except queue.Full:
                pass

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.queue.put(FRAME_SENTINEL)
            self.thread.join()

    def _write_loop(self):
        while True:
            item = self.queue.get()
            if item is FRAME_SENTINEL:
                break
            path, frame = item
            cv2.imwrite(str(path), frame)
