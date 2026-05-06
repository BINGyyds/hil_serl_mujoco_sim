__all__ = ["RobotController", "TaskSpaceVelocityController", "TeleopRuntime"]


def __getattr__(name):
    if name == "RobotController":
        from serl_robot_infra.control.robot_controller import RobotController

        return RobotController
    if name == "TaskSpaceVelocityController":
        from serl_robot_infra.control.task_space_velocity_controller import (
            TaskSpaceVelocityController,
        )

        return TaskSpaceVelocityController
    if name == "TeleopRuntime":
        from serl_robot_infra.control.teleop_runtime import TeleopRuntime

        return TeleopRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
