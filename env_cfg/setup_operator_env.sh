#!/usr/bin/env bash
# setup_4090_operator_env.sh
# 用途：新租 RTX 4090 / NVIDIA GPU 服务器后，一次性配置算子开发环境。
# 适用：Ubuntu 22.04 / 24.04，云主机通常已安装 NVIDIA 驱动。
#
# 默认安装内容：
# - apt 基础工具：git/cmake/ninja/build-essential/python3-venv 等
# - 可选 CUDA Toolkit 12.6：用于 nvcc 编译 PyTorch C++/CUDA Extension
# - Python venv：默认放在 $HOME/.venvs/oplab
# - PyTorch CUDA wheel：默认 cu126
# - Python 开发包：pytest/numpy/pandas/matplotlib/ninja/pybind11 等
# - Nsight Systems / Nsight Compute：如果 NVIDIA apt repo 可用则安装
# - 项目仓库：默认 public HTTPS clone，可无 GitHub 认证下载
#
# 常用执行：
#   bash setup_4090_operator_env.sh
#
# 指定公开仓库：
#   REPO_URL=https://github.com/zhangxiao8809095/ai_operator_path.git bash setup_4090_operator_env.sh
#
# 如果云主机没有 nvcc，允许脚本安装 CUDA Toolkit：
#   INSTALL_CUDA_TOOLKIT=1 bash setup_4090_operator_env.sh
#
# 如果只想配置环境，不 clone 项目：
#   CLONE_REPO=0 bash setup_4090_operator_env.sh

set -Eeuo pipefail

# ========== 可配置参数 ==========
REPO_URL="${REPO_URL:-https://github.com/zhangxiao8809095/ai_operator_path.git}"
PROJECT_PARENT="${PROJECT_PARENT:-$HOME/projects}"
PROJECT_NAME="${PROJECT_NAME:-ai_operator_path}"
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_PARENT/$PROJECT_NAME}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/oplab}"

# PyTorch CUDA wheel。可改成 cu128 / cu126 / cu118。
TORCH_CUDA="${TORCH_CUDA:-cu126}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA}}"

# RTX 4090 Ada Lovelace compute capability = 8.9
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

# 是否安装 CUDA Toolkit。如果服务器已有 nvcc，通常不需要。
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-0}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12-6}"

# 是否 clone / pull 项目
CLONE_REPO="${CLONE_REPO:-1}"

# 是否安装 Nsight 命令行工具
INSTALL_NSIGHT="${INSTALL_NSIGHT:-1}"

# ========== 日志工具 ==========
log()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m $*" >&2; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ========== 0. 基础系统检查 ==========
log "Start RTX 4090 operator development environment setup"

if [[ ! -f /etc/os-release ]]; then
  err "Only Linux with /etc/os-release is supported."
  exit 1
fi

source /etc/os-release
log "Detected OS: ${PRETTY_NAME:-unknown}"

if [[ "${ID:-}" != "ubuntu" ]]; then
  warn "This script is mainly tested for Ubuntu. Current OS ID=${ID:-unknown}."
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  warn "Current architecture is $ARCH. Most RTX 4090 cloud servers should be x86_64."
fi

# ========== 1. apt 基础依赖 ==========
log "Installing base apt packages"
run_sudo apt-get update
run_sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  git \
  git-lfs \
  cmake \
  ninja-build \
  pkg-config \
  curl \
  wget \
  ca-certificates \
  gnupg \
  lsb-release \
  software-properties-common \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  unzip \
  zip \
  htop \
  tmux \
  tree \
  rsync \
  jq

# ========== 2. NVIDIA GPU / Driver 检查 ==========
log "Checking NVIDIA driver and GPU"

if need_cmd nvidia-smi; then
  nvidia-smi || true
else
  err "nvidia-smi not found. The NVIDIA driver is missing or not visible."
  err "建议换一个已预装 NVIDIA driver/CUDA 的 4090 镜像；不要在租赁机器上盲目重装驱动。"
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
log "Detected GPU: ${GPU_NAME:-unknown}"

# ========== 3. CUDA Toolkit / nvcc 检查 ==========
log "Checking nvcc"

if need_cmd nvcc; then
  nvcc --version || true
else
  warn "nvcc not found. PyTorch CUDA runtime may still work, but custom CUDA extension needs nvcc."

  if [[ "$INSTALL_CUDA_TOOLKIT" == "1" ]]; then
    log "INSTALL_CUDA_TOOLKIT=1, installing CUDA Toolkit ${CUDA_TOOLKIT_VERSION}"

    UBUNTU_VER="$(. /etc/os-release && echo "${VERSION_ID//./}")"
    CUDA_REPO="ubuntu${UBUNTU_VER}"

    if [[ "$CUDA_REPO" != "ubuntu2204" && "$CUDA_REPO" != "ubuntu2404" && "$CUDA_REPO" != "ubuntu2004" ]]; then
      err "Unsupported Ubuntu version for automatic CUDA repo setup: $CUDA_REPO"
      err "Please install CUDA Toolkit manually or use a cloud image with nvcc preinstalled."
      exit 1
    fi

    TMP_DEB="/tmp/cuda-keyring_1.1-1_all.deb"
    wget -O "$TMP_DEB" "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    run_sudo dpkg -i "$TMP_DEB"
    run_sudo apt-get update
    run_sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "cuda-toolkit-${CUDA_TOOLKIT_VERSION}"

    if [[ -d "/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}" ]]; then
      CUDA_HOME_DEFAULT="/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}"
    elif [[ -d "/usr/local/cuda" ]]; then
      CUDA_HOME_DEFAULT="/usr/local/cuda"
    else
      CUDA_HOME_DEFAULT=""
    fi

    if [[ -n "$CUDA_HOME_DEFAULT" ]]; then
      export CUDA_HOME="$CUDA_HOME_DEFAULT"
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi

    nvcc --version || {
      err "CUDA Toolkit installation attempted, but nvcc still not found."
      exit 1
    }
  else
    warn "Skip CUDA Toolkit installation. To install toolkit automatically, run:"
    warn "  INSTALL_CUDA_TOOLKIT=1 bash setup_4090_operator_env.sh"
  fi
fi

# 写入 shell 配置，保证重新登录后可用
CUDA_CANDIDATE=""
if need_cmd nvcc; then
  CUDA_CANDIDATE="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [[ -d "/usr/local/cuda" ]]; then
  CUDA_CANDIDATE="/usr/local/cuda"
fi

if [[ -n "$CUDA_CANDIDATE" ]]; then
  log "Setting CUDA_HOME=${CUDA_CANDIDATE}"
  {
    echo ""
    echo "# Operator lab CUDA environment"
    echo "export CUDA_HOME=${CUDA_CANDIDATE}"
    echo 'export PATH="$CUDA_HOME/bin:$PATH"'
    echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"'
    echo 'export TORCH_CUDA_ARCH_LIST="8.9"'
  } >> "$HOME/.bashrc"
fi

# ========== 4. Python venv ==========
log "Creating Python venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading pip/setuptools/wheel"
python -m pip install -U pip setuptools wheel

# ========== 5. 安装 PyTorch CUDA 版 ==========
log "Installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio

# ========== 6. 安装算子开发常用 Python 包 ==========
log "Installing Python development packages"
python -m pip install -U \
  numpy \
  pandas \
  matplotlib \
  pytest \
  pytest-xdist \
  ninja \
  pybind11 \
  packaging \
  tqdm \
  tabulate \
  ipython \
  jupyterlab

# ========== 7. 安装 Nsight 命令行工具 ==========
if [[ "$INSTALL_NSIGHT" == "1" ]]; then
  log "Installing Nsight Systems / Nsight Compute if available"

  run_sudo apt-get update || true

  if apt-cache show nsight-systems >/dev/null 2>&1; then
    run_sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nsight-systems || warn "Failed to install nsight-systems"
  else
    warn "nsight-systems package not found in current apt repos."
  fi

  # Common package names vary by repo/version. Try broad names first.
  if apt-cache search '^nsight-compute' | grep -q nsight-compute; then
    NSIGHT_COMPUTE_PKG="$(apt-cache search '^nsight-compute' | awk '{print $1}' | head -n 1)"
    if [[ -n "$NSIGHT_COMPUTE_PKG" ]]; then
      run_sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$NSIGHT_COMPUTE_PKG" || warn "Failed to install $NSIGHT_COMPUTE_PKG"
    fi
  else
    warn "nsight-compute package not found in current apt repos."
    warn "If ncu is missing, use a CUDA Toolkit image or install Nsight Compute manually from NVIDIA."
  fi
fi

# ========== 8. clone / update 项目 ==========
if [[ "$CLONE_REPO" == "1" ]]; then
  log "Preparing project directory: $PROJECT_DIR"
  mkdir -p "$PROJECT_PARENT"

  if [[ -d "$PROJECT_DIR/.git" ]]; then
    log "Project already exists. Pulling latest code."
    git -C "$PROJECT_DIR" pull --ff-only || warn "git pull failed. You can inspect manually: cd $PROJECT_DIR && git status"
  elif [[ -e "$PROJECT_DIR" ]]; then
    warn "$PROJECT_DIR exists but is not a git repo. Skip clone."
  else
    log "Cloning repo: $REPO_URL"
    git clone "$REPO_URL" "$PROJECT_DIR" || warn "git clone failed. If repo is private, make it public or use rsync."
  fi
fi

# ========== 9. 生成便捷脚本 ==========
BIN_DIR="$HOME/bin"
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/oplab_activate" <<EOF
#!/usr/bin/env bash
source "$VENV_DIR/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.9"
if [[ -d "$PROJECT_DIR" ]]; then
  cd "$PROJECT_DIR"
fi
echo "Activated operator lab env: $VENV_DIR"
echo "Project dir: $PROJECT_DIR"
EOF
chmod +x "$BIN_DIR/oplab_activate"

if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
fi

# ========== 10. 自检 ==========
log "Running environment self-check"

python - <<'PY'
import os
import torch

print("Python:", os.sys.version.replace("\n", " "))
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Matmul OK:", float(y[0, 0]))
else:
    raise SystemExit("torch.cuda.is_available() is False")
PY

log "Tool versions"
echo "git: $(git --version || true)"
echo "cmake: $(cmake --version | head -n 1 || true)"
echo "ninja: $(ninja --version || true)"
echo "python: $(python --version || true)"
echo "nvcc: $(nvcc --version | tail -n 1 || echo 'not found')"
echo "nsys: $(nsys --version 2>/dev/null || echo 'not found')"
echo "ncu: $(ncu --version 2>/dev/null | head -n 1 || echo 'not found')"

# ========== 11. 项目编译测试提示 ==========
if [[ -d "$PROJECT_DIR" ]]; then
  log "Project is ready at: $PROJECT_DIR"
  cat <<EOF

Next commands:

  source "$VENV_DIR/bin/activate"
  cd "$PROJECT_DIR"

  # 如果项目是 PyTorch C++/CUDA Extension：
  pip install -e .

  # 正确性测试：
  pytest -q

  # benchmark：
  python benchmark/bench_ops.py

  # Nsight 示例：
  bash scripts/profile_ncu.sh gemm_tiled
  bash scripts/profile_nsys.sh

EOF
fi

log "Setup completed."
log "以后登录服务器后可以执行：oplab_activate"
