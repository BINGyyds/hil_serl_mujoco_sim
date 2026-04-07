#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sim_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"
CONDA_CHANNEL_ARGS=(--override-channels -c conda-forge -c defaults)

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

if [[ "$(uname)" != "Darwin" ]]; then
    error "setup_env_mac.sh 仅适用于 macOS。"
fi

if ! command -v brew &>/dev/null; then
    error "未找到 Homebrew。请先安装 Homebrew，再运行本脚本。"
fi

if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
elif [[ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]]; then
    CONDA_BIN="/opt/homebrew/Caskroom/miniconda/base/bin/conda"
else
    error "未找到 Conda。"
fi

eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" \
    || eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)" \
    || true

info "Conda 安装将忽略本机 .condarc，自带强制使用官方 channels: conda-forge + defaults"

if conda env list | grep -qw "$ENV_NAME"; then
    warn "Conda 环境 '$ENV_NAME' 已存在，跳过创建。"
else
    info "创建 Conda 环境: $ENV_NAME"
    conda create "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

if brew list hidapi &>/dev/null; then
    info "检测到 Homebrew hidapi，跳过安装。"
else
    info "通过 Homebrew 安装 hidapi"
    brew install hidapi
fi

info "升级 pip / setuptools / wheel"
python -m pip install --upgrade pip setuptools wheel

info "安装 Pinocchio"
conda install "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" pinocchio -y \
    || warn "Pinocchio 安装失败；如果后续运行 robot.main 报错，请单独处理 conda-forge pinocchio。"

info "以 editable 方式安装 robot_controllers_mujoco"
python -m pip install -e "$PROJECT_DIR"

if [[ "$ARCH" == "arm64" ]]; then
    info "Apple Silicon: 替换为 patched easyhid fork"
    python -m pip uninstall -y easyhid easyhid-ng >/dev/null 2>&1 || true
    python -m pip install --force-reinstall git+https://github.com/bglopez/python-easyhid.git
else
    info "Intel macOS: 保留默认 easyhid 安装"
fi

info "验证 insertion Gym 环境"
python -c "
from robot_controllers_mujoco.configs.insertion_config import make_insertion_config
from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv
env = FrankaInsertionEnv(config=make_insertion_config())
obs, _ = env.reset()
print('obs keys:', list(obs['state'].keys()))
print('action space:', env.action_space)
env.close()
"

info "验证核心 Python 依赖"
python -c "
import cv2
import matplotlib
import mujoco
import numpy as np
import pandas as pd
import pinocchio as pin
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('matplotlib:', matplotlib.__version__)
print('mujoco:', mujoco.__version__)
print('opencv:', cv2.__version__)
print('pinocchio:', pin.__version__)
" || warn "核心依赖导入失败；请检查 pip/conda 安装日志。"

info "验证 SpaceMouse 模块导入"
python -c "from robot_controllers_mujoco.spacemouse.spacemouse_expert import SpaceMouseExpert; print('SpaceMouse import ok')" \
    || warn "SpaceMouse 模块导入失败；如果你不使用 SpaceMouse，可暂时忽略。"

echo
info "环境已准备完成。"
info "激活环境: conda activate $ENV_NAME"
info "控制器入口: python -m robot_controllers_mujoco.robot.main --input-mode spacemouse --control-mode velocity --allow-missing-spacemouse"
info "录制 demo: python scripts/record_demos_mujoco.py --output demo_data/insertion_demos.pkl --num_demos 25"
warn "如果安装了 3Dconnexion 官方驱动，运行前建议先退出 3DconnexionHelper。"
