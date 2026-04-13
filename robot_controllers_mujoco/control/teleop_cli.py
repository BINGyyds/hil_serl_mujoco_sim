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
    parser = argparse.ArgumentParser(description="机器人控制与仿真框架")
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="禁用实时可视化界面，以无头模式运行仿真。",
    )
    parser.add_argument(
        "--input-mode",
        choices=("planned", "spacemouse"),
        default=None,
        help="输入模式：planned 使用内置轨迹规划，spacemouse 使用 6D SpaceMouse 实时控制。",
    )
    parser.add_argument(
        "--control-mode",
        choices=("position", "effort", "velocity"),
        default=None,
        help="底层控制模式。",
    )
    parser.add_argument(
        "--allow-missing-spacemouse",
        action="store_true",
        help="当 SpaceMouse 未连接或本机 HID 后端不可用时，回退到零输入 mock teleop 模式。",
    )

    args = parser.parse_args()
    logger.info("--- 启动机器人控制项目 ---")

    try:
        config = ProjectConfig.from_overrides(
            use_viewer=not args.no_viewer,
            input_mode=args.input_mode,
            control_mode=args.control_mode,
            allow_missing_spacemouse=args.allow_missing_spacemouse,
        )
        controller = RobotController(config)
        controller.run_simulation()
    except Exception as exc:
        logger.exception("仿真过程中发生未处理的异常: %s", exc)

    logger.info("--- 项目执行完毕 ---")
