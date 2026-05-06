__all__ = ["AsyncVideoWriter", "DataManager", "ProjectConfig"]


def __getattr__(name):
    if name == "AsyncVideoWriter":
        from serl_robot_infra.common.async_writer import AsyncVideoWriter

        return AsyncVideoWriter
    if name == "DataManager":
        from serl_robot_infra.common.data_manager import DataManager

        return DataManager
    if name == "ProjectConfig":
        from serl_robot_infra.common.config import ProjectConfig

        return ProjectConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
