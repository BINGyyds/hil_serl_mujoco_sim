from importlib import import_module


__all__ = [
    "AsyncCameraWindow",
    "AsyncVideoWriter",
    "DataManager",
    "ProjectConfig",
    "se3_gic",
    "se3_utils",
]


def __getattr__(name):
    if name == "AsyncCameraWindow":
        from serl_robot_infra.common.async_camera_window import AsyncCameraWindow

        return AsyncCameraWindow
    if name == "AsyncVideoWriter":
        from serl_robot_infra.common.async_writer import AsyncVideoWriter

        return AsyncVideoWriter
    if name == "DataManager":
        from serl_robot_infra.common.data_manager import DataManager

        return DataManager
    if name == "ProjectConfig":
        from serl_robot_infra.common.config import ProjectConfig

        return ProjectConfig
    if name == "se3_gic":
        return import_module("serl_robot_infra.common.se3_gic")
    if name == "se3_utils":
        return import_module("serl_robot_infra.common.se3_utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
