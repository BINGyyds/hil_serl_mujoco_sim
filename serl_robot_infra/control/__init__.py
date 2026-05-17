__all__ = [
    "RobotController",
    "SE3TorqueController",
    "NRICTorqueController",
    "SE3TorqueRuntime",
    "TaskSpaceVelocityController",
    "TeleopRuntime",
]


def __getattr__(name):
    if name == "RobotController":
        from serl_robot_infra.control.robot_controller import RobotController

        return RobotController
    if name == "SE3TorqueController":
        from serl_robot_infra.control.se3_torque_controller import SE3TorqueController

        return SE3TorqueController
    if name == "NRICTorqueController":
        from serl_robot_infra.control.nric_torque_controller import (
            NRICTorqueController,
        )

        return NRICTorqueController
    if name == "SE3TorqueRuntime":
        from serl_robot_infra.control.se3_torque_runtime import SE3TorqueRuntime

        return SE3TorqueRuntime
    if name == "TaskSpaceVelocityController":
        from serl_robot_infra.control.task_space_velocity_controller import (
            TaskSpaceVelocityController,
        )

        return TaskSpaceVelocityController
    if name == "TeleopRuntime":
        from serl_robot_infra.control.teleop_runtime import TeleopRuntime

        return TeleopRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
