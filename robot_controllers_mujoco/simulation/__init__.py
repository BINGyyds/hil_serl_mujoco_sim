__all__ = [
    "MujocoInterface",
    "RobotModel",
    "RobotMotion",
    "SiteIKOptions",
    "SiteIKResult",
    "solve_site_pose_ik",
]


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
    if name in {"SiteIKOptions", "SiteIKResult", "solve_site_pose_ik"}:
        from robot_controllers_mujoco.simulation import site_ik_solver

        return getattr(site_ik_solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
