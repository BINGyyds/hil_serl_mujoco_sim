"""Top-level package for the local MuJoCo simulation repository."""

from importlib import import_module
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "models"


def get_franka_model_path(model_name: str = "insertion_scene.xml") -> Path:
    """Return the absolute path to a Franka MJCF file shipped with the package."""
    return MODEL_ROOT / "franka_panda" / model_name


def __getattr__(name: str):
    if name in {
        "InsertionConfig",
        "make_insertion_config",
    }:
        module = import_module("serl_robot_infra.configs")
        return getattr(module, name)
    if name in {
        "FrankaMujocoEnv",
        "FrankaInsertionEnv",
        "MujocoSERLObsWrapper",
        "MujocoSpacemouseIntervention",
    }:
        module = import_module("serl_robot_infra.envs")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FrankaMujocoEnv",
    "FrankaInsertionEnv",
    "MujocoSERLObsWrapper",
    "MujocoSpacemouseIntervention",
    "InsertionConfig",
    "make_insertion_config",
    "get_franka_model_path",
]
