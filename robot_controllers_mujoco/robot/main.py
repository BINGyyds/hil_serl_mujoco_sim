import argparse
import logging
import sys
from pathlib import Path

# Allow direct script execution from the repo checkout; the standard entry
# point remains `python -m robot_controllers_mujoco.robot.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_controllers_mujoco.robot.config import ProjectConfig
from robot_controllers_mujoco.robot.robot_controller import RobotController

# 配置日志系统，这是程序的第一个动作
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    程序主入口函数。
    
    负责以下核心任务:
    1. 解析命令行参数，允许用户在运行时覆盖默认配置。
    2. 基于默认配置和命令行参数，创建一个全局的 `ProjectConfig` 配置实例。
    3. 初始化核心的 `RobotController`。
    4. 启动并运行仿真。
    5. 捕获并记录任何未处理的顶层异常，确保程序健壮性。
    """
    # 1. 创建命令行参数解析器，定义可配置的运行时选项
    parser = argparse.ArgumentParser(description="机器人控制与仿真框架")

    parser.add_argument(
        '--no_viewer', 
        action='store_true',
        help='禁用实时可视化界面，以无头模式运行仿真。'
    )

    # 解析传入的命令行参数
    args = parser.parse_args()

    logger.info("--- 启动机器人控制项目 ---")
    
    try:
        # 2. 创建配置对象，并使用命令行参数覆盖默认值
        # 这是整个项目配置的唯一来源
        config = ProjectConfig(
            USE_VIEWER=not args.no_viewer,
        )
        
        # 3. 创建控制器实例并传入配置对象
        controller = RobotController(config)
        
        # 4. 运行主仿真
        controller.run_simulation()
        
    except Exception as e:
        # 5. 捕获任何未处理的异常并记录详细信息
        logger.exception(f"仿真过程中发生未处理的异常: {e}")

    logger.info("--- 项目执行完毕 ---")

if __name__ == '__main__':
    main()
