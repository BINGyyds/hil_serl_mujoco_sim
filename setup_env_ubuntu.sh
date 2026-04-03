#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sim_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_APT="${SKIP_APT:-0}"
CONDA_CHANNEL_ARGS=(--override-channels -c conda-forge -c defaults)

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

if [[ "$(uname)" != "Linux" ]]; then
    error "setup_env_ubuntu.sh 仅适用于 Ubuntu/Linux。"
fi

if ! command -v apt-get &>/dev/null; then
    error "未找到 apt-get。当前脚本面向 Ubuntu/Debian 系统。"
fi

if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
else
    error "未找到 Conda。"
fi

eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" \
    || eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)" \
    || true

info "Conda 安装将忽略本机 .condarc，自带强制使用官方 channels: conda-forge + defaults"

if [[ "$SKIP_APT" == "1" ]]; then
    warn "已设置 SKIP_APT=1，跳过 Ubuntu 系统依赖安装。"
else
    info "安装 Ubuntu 系统依赖"
    if ! sudo apt-get update; then
        error "apt-get update 失败。通常是系统里存在失效的第三方源。请先修复 /etc/apt/sources.list.d 下的坏源后重试；如果这些系统依赖已经装好，也可以使用 SKIP_APT=1 ./setup_env_ubuntu.sh 跳过此步骤。"
    fi

    sudo apt-get install -y \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libhidapi-dev \
        libhidapi-hidraw0 \
        libsm6 \
        libxext6 \
        libxrender1
fi

if conda env list | grep -qw "$ENV_NAME"; then
    warn "Conda 环境 '$ENV_NAME' 已存在，跳过创建。"
else
    info "创建 Conda 环境: $ENV_NAME"
    conda create "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

info "升级 pip / setuptools / wheel"
python -m pip install --upgrade pip setuptools wheel

info "安装 Pinocchio"
conda install "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" pinocchio -y \
    || warn "Pinocchio 安装失败；如果后续运行 robot.main 报错，请单独处理 conda-forge pinocchio。"

info "以 editable 方式安装 sim_env"
python -m pip install -e "$PROJECT_DIR"

info "验证 Gym 环境"
python -c "
from robot_controllers_mujoco import FrankaReachEnv, make_reach_config
env = FrankaReachEnv(config=make_reach_config())
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
info "录制 demo: python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl"
warn "如需普通用户访问 SpaceMouse，请安装当前目录下的 99-spacemouse.rules。"
