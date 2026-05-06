#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sim_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_APT="${SKIP_APT:-0}"
SKIP_UDEV="${SKIP_UDEV:-0}"
APT_LOCK_TIMEOUT="${APT_LOCK_TIMEOUT:-300}"
CONDA_CHANNEL_ARGS=(--override-channels -c conda-forge)

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

if [[ "$(uname)" != "Linux" ]]; then
    error "setup_env_sim.sh 仅适用于 Ubuntu/Linux。"
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

info "Conda 安装将忽略本机 .condarc，只使用 conda-forge，避免 defaults channel 的 Anaconda ToS 阻塞"

if [[ "$SKIP_APT" == "1" ]]; then
    warn "已设置 SKIP_APT=1，跳过 Ubuntu 系统依赖安装。"
else
    info "安装 Ubuntu 系统依赖"
    if ! sudo apt-get -o DPkg::Lock::Timeout="$APT_LOCK_TIMEOUT" update; then
        error "apt-get update 失败。通常是系统里存在失效的第三方源。请先修复 /etc/apt/sources.list.d 下的坏源后重试；如果这些系统依赖已经装好，也可以使用 SKIP_APT=1 ./setup_env_sim.sh 跳过此步骤。"
    fi

    sudo apt-get -o DPkg::Lock::Timeout="$APT_LOCK_TIMEOUT" install -y \
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

if [[ "$SKIP_UDEV" == "1" ]]; then
    warn "已设置 SKIP_UDEV=1，跳过 SpaceMouse udev 规则安装。"
elif [[ -f "$PROJECT_DIR/99-spacemouse.rules" ]]; then
    info "安装 SpaceMouse udev 规则"
    sudo install -m 0644 "$PROJECT_DIR/99-spacemouse.rules" /etc/udev/rules.d/99-spacemouse.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=hidraw || warn "udevadm trigger 失败；请拔插 SpaceMouse 后重试。"
else
    warn "未找到 99-spacemouse.rules，跳过 SpaceMouse udev 规则安装。"
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
    || warn "Pinocchio 安装失败；如果后续运行 teleop/debug 入口报错，请单独处理 conda-forge pinocchio。"

info "以 editable 方式安装 sim_env"
python -m pip install -e "$PROJECT_DIR"

info "验证 insertion Gym 环境"
python -c "
from serl_robot_infra.configs.insertion_config import make_insertion_config
from serl_robot_infra.envs.franka_insertion_env import FrankaInsertionEnv
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
python -c "from serl_robot_infra.spacemouse.spacemouse_expert import SpaceMouseExpert; print('SpaceMouse import ok')" \
    || warn "SpaceMouse 模块导入失败；如果你不使用 SpaceMouse，可暂时忽略。"

echo
info "环境已准备完成。"
info "激活环境: conda activate $ENV_NAME"
info "控制器入口: python scripts/debug/teleop_main.py --allow-missing-spacemouse"
info "录制 demo: python scripts/record_demos_mujoco.py --output demo_data/insertion_demos.pkl --num_demos 25"
warn "如果 SpaceMouse 已经插着但仍无法打开，请拔插一次设备后重试。"
