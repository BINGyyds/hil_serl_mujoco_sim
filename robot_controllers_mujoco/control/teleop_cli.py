import argparse
import logging

from robot_controllers_mujoco.common.config import ProjectConfig
from robot_controllers_mujoco.control.robot_controller import RobotController


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the manual MuJoCo control/debug entrypoint."""
    parser = argparse.ArgumentParser(
        description="Franka insertion 的 SpaceMouse + velocity teleop 调试入口"
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="禁用实时可视化界面，以无头模式运行仿真。",
    )
    parser.add_argument(
        "--input-mode",
        choices=("spacemouse",),
        default=None,
        help="当前仅保留 SpaceMouse 实时控制路径。",
    )
    parser.add_argument(
        "--control-mode",
        choices=("velocity",),
        default=None,
        help="当前仅保留 velocity 控制模式。",
    )
    parser.add_argument(
        "--allow-missing-spacemouse",
        action="store_true",
        help="当 SpaceMouse 未连接或本机 HID 后端不可用时，回退到零输入 mock teleop 模式。",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="打开末端左右相机实时拼接窗口。",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=20,
        help="末端相机窗口刷新率。",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=320,
        help="每路末端相机画面宽度。",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=240,
        help="每路末端相机画面高度。",
    )

    args = parser.parse_args()
    logger.info("--- 启动机器人控制项目 ---")

    try:
        config = ProjectConfig.from_overrides(
            use_viewer=not args.no_viewer,
            input_mode=args.input_mode,
            control_mode=args.control_mode,
            allow_missing_spacemouse=args.allow_missing_spacemouse,
            show_teleop_cameras=args.show_cameras,
            teleop_camera_fps=args.camera_fps,
            teleop_camera_width=args.camera_width,
            teleop_camera_height=args.camera_height,
        )
        controller = RobotController(config)
        controller.run_simulation()
    except Exception as exc:
        logger.exception("仿真过程中发生未处理的异常: %s", exc)

    logger.info("--- 项目执行完毕 ---")
