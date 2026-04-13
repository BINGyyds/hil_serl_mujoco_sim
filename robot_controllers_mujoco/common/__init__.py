__all__ = ["AsyncVideoWriter", "DataManager", "ProjectConfig"]


def __getattr__(name):
    if name == "AsyncVideoWriter":
        from robot_controllers_mujoco.common.async_writer import AsyncVideoWriter

        return AsyncVideoWriter
    if name == "DataManager":
        from robot_controllers_mujoco.common.data_manager import DataManager

        return DataManager
    if name == "ProjectConfig":
        from robot_controllers_mujoco.common.config import ProjectConfig

        return ProjectConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
