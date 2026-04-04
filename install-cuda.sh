#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# live-transcript CUDA installation script
# Target: Linux + NVIDIA GPU
# Installs miniconda (if needed), CUDA toolkit via conda,
# and sherpa-onnx CUDA wheel.
# ============================================================

CONDA_ENV_NAME="live-transcript"
PYTHON_VERSION="3.11"
CUDA_TOOLKIT_VERSION="12.6"
SHERPA_ONNX_VERSION="1.12.35"
SHERPA_CUDA_SUFFIX="+cuda12.cudnn9"
CUDA_WHEEL_INDEX="https://k2-fsa.github.io/sherpa/onnx/cuda.html"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_DIR="$HOME/miniconda3"

# --- Color helpers ----------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# --- Check GPU --------------------------------------------------------------
info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. NVIDIA driver is required."
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ok "NVIDIA GPU detected"

# --- Install miniconda if needed --------------------------------------------
install_miniconda() {
    if command -v conda &>/dev/null; then
        ok "conda already installed: $(conda --version)"
        return
    fi

    if [ -x "$MINICONDA_DIR/bin/conda" ]; then
        ok "miniconda found at $MINICONDA_DIR (not in PATH yet)"
        eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
        return
    fi

    info "Installing miniconda..."
    local installer="/tmp/miniconda-installer.sh"
    wget -q -O "$installer" "$MINICONDA_URL"
    bash "$installer" -b -p "$MINICONDA_DIR"
    rm -f "$installer"

    eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
    conda init --quiet
    ok "miniconda installed to $MINICONDA_DIR"
}

install_miniconda

# Ensure conda is available in current shell
if ! command -v conda &>/dev/null; then
    eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
fi

# --- Create or reuse conda environment -------------------------------------
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    info "Conda environment '$CONDA_ENV_NAME' already exists, activating..."
    conda activate "$CONDA_ENV_NAME"
else
    info "Creating conda environment '$CONDA_ENV_NAME' (Python $PYTHON_VERSION + CUDA $CUDA_TOOLKIT_VERSION)..."
    conda create -n "$CONDA_ENV_NAME" \
        python="$PYTHON_VERSION" \
        "cuda-toolkit=$CUDA_TOOLKIT_VERSION" \
        cudnn \
        -c nvidia -y
    conda activate "$CONDA_ENV_NAME"
    ok "Conda environment created and activated"
fi

# --- Verify CUDA runtime ----------------------------------------------------
info "Verifying CUDA runtime libraries..."
python -c "
import ctypes, os, sys

for lib in ['libcublas.so', 'libcublasLt.so', 'libcudnn.so']:
    try:
        ctypes.CDLL(lib)
        print(f'  {lib}: OK')
    except OSError:
        print(f'  {lib}: NOT FOUND', file=sys.stderr)
        sys.exit(1)
"
ok "CUDA runtime libraries available"

# --- Install project --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

info "Installing live-transcript from $PROJECT_DIR..."
pip install -e .

# --- Install sherpa-onnx CUDA wheel ----------------------------------------
info "Installing sherpa-onnx ${SHERPA_ONNX_VERSION}${SHERPA_CUDA_SUFFIX}..."
pip install "sherpa-onnx==${SHERPA_ONNX_VERSION}${SHERPA_CUDA_SUFFIX}" \
    -f "$CUDA_WHEEL_INDEX"

# --- Download models --------------------------------------------------------
info "Downloading models..."
python scripts/download_models.py

# --- Final verification -----------------------------------------------------
info "Verifying installation..."
python -c "
import sherpa_onnx
print(f'  sherpa-onnx version: {sherpa_onnx.__version__}')
"
ok "Installation complete!"

echo ""
echo "============================================================"
echo "  To start the server:"
echo ""
echo "    conda activate $CONDA_ENV_NAME"
echo "    python -m live_transcript -c config_cuda.yaml"
echo ""
echo "============================================================"
