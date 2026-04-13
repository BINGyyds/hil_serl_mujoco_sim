__all__ = ["MujocoInterface", "RobotModel", "RobotMotion"]


def __getattr__(name):
    if name == "MujocoInterface":
        from robot_controllers_mujoco.simulation.mujoco_interface import MujocoInterface

        return MujocoInterface
    if name == "RobotModel":
        from robot_controllers_mujoco.simulation.robot_model import RobotModel

        return RobotModel
    if name == "RobotMotion":
        from robot_controllers_mujoco.simulation.robot_motion import RobotMotion

        return RobotMotion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
