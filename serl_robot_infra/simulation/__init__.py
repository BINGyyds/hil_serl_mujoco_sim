__all__ = [
    "MujocoInterface",
    "SiteIKOptions",
    "SiteIKResult",
    "solve_site_pose_ik",
]


def __getattr__(name):
    if name == "MujocoInterface":
        from serl_robot_infra.simulation.mujoco_interface import MujocoInterface

        return MujocoInterface
    if name in {"SiteIKOptions", "SiteIKResult", "solve_site_pose_ik"}:
        from serl_robot_infra.simulation import site_ik_solver

        return getattr(site_ik_solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
