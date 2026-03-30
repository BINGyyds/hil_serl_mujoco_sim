from importlib import import_module

__all__ = ["SpaceMouseExpert", "ZeroSpaceMouseExpert"]


def __getattr__(name: str):
    if name == "SpaceMouseExpert":
        return import_module(
            "robot_controllers_mujoco.spacemouse.spacemouse_expert"
        ).SpaceMouseExpert
    if name == "ZeroSpaceMouseExpert":
        return import_module(
            "robot_controllers_mujoco.spacemouse.spacemouse_expert"
        ).ZeroSpaceMouseExpert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
