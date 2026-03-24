"""Top-level package for the local MuJoCo simulation repository."""

from importlib import import_module
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "models"


def get_franka_model_path(model_name: str = "reach_scene.xml") -> Path:
    """Return the absolute path to a Franka MJCF file shipped with the package."""
    return MODEL_ROOT / "franka_panda" / model_name


def __getattr__(name: str):
    if name in {"ReachConfig", "TrainConfig", "make_reach_config"}:
        module = import_module("robot_controllers_mujoco.configs")
        return getattr(module, name)
    if name in {
        "FrankaMujocoEnv",
        "FrankaReachEnv",
        "MujocoSERLObsWrapper",
        "MujocoSpacemouseIntervention",
    }:
        module = import_module("robot_controllers_mujoco.envs")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FrankaMujocoEnv",
    "FrankaReachEnv",
    "MujocoSERLObsWrapper",
    "MujocoSpacemouseIntervention",
    "ReachConfig",
    "make_reach_config",
    "get_franka_model_path",
]
