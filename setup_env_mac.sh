#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-robot_controllers_mujoco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

if [[ "$(uname)" != "Darwin" ]]; then
    error "setup_env_mac.sh 仅适用于 macOS。"
fi

if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
    CONDA_BIN="/opt/homebrew/Caskroom/miniconda/base/bin/conda"
else
    error "未找到 Conda。"
fi

eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" || eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)" || true

if conda env list | grep -qw "$ENV_NAME"; then
    warn "Conda 环境 '$ENV_NAME' 已存在，跳过创建。"
else
    info "创建 Conda 环境: $ENV_NAME"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

info "升级 pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel

info "安装 robot_controllers_mujoco"
pip install -e "$PROJECT_DIR"

info "安装可选的本地调试依赖"
pip install stable-baselines3 || warn "stable-baselines3 安装失败，可忽略。"

info "验证 MuJoCo 环境"
python -c "
from robot_controllers_mujoco import FrankaReachEnv, make_reach_config
env = FrankaReachEnv(config=make_reach_config())
obs, _ = env.reset()
print('obs keys:', list(obs['state'].keys()))
print('action space:', env.action_space)
env.close()
"

info "验证 SpaceMouse 模块导入"
python -c "from robot_controllers_mujoco.spacemouse.spacemouse_expert import SpaceMouseExpert; print('SpaceMouse import ok')" \
    || warn "SpaceMouse 模块导入失败；如果你不使用 SpaceMouse，可暂时忽略。"

echo
info "环境已准备完成。"
info "激活环境: conda activate $ENV_NAME"
info "录制 demo: python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl"
warn "如果后续要运行 distributed actor，还需要安装 train/ 侧的 actor 环境。"
