#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# live-transcript CUDA installation script
# Target: Linux + NVIDIA GPU (RTX 4090 class)
# ============================================================

SHERPA_ONNX_VERSION="1.12.35"
CUDA_WHEEL_INDEX="https://k2-fsa.github.io/sherpa/onnx/cuda.html"

# --- Detect CUDA version ---------------------------------------------------
detect_cuda() {
    if command -v nvcc &>/dev/null; then
        local ver
        ver=$(nvcc --version | grep -oP 'release \K[0-9]+')
        echo "$ver"
    elif command -v nvidia-smi &>/dev/null; then
        local ver
        ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
        echo "$ver"
    else
        echo ""
    fi
}

CUDA_MAJOR=$(detect_cuda)

if [ -z "$CUDA_MAJOR" ]; then
    echo "ERROR: CUDA not found. Install NVIDIA CUDA Toolkit first."
    echo "  https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "Detected CUDA major version: ${CUDA_MAJOR}"

if [ "$CUDA_MAJOR" -ge 12 ]; then
    SHERPA_SUFFIX="+cuda12.cudnn9"
    echo "Using sherpa-onnx CUDA 12 + cuDNN 9 wheel"
else
    SHERPA_SUFFIX="+cuda"
    echo "Using sherpa-onnx CUDA 11.8 + cuDNN 8 wheel"
fi

# --- Create venv if not active ---------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "Using existing virtual environment: $VIRTUAL_ENV"
fi

# --- Install project --------------------------------------------------------
echo ""
echo "Installing live-transcript..."
pip install -e .

# --- Install sherpa-onnx CUDA wheel (overrides CPU version) ----------------
echo ""
echo "Installing sherpa-onnx ${SHERPA_ONNX_VERSION}${SHERPA_SUFFIX}..."
pip install "sherpa-onnx==${SHERPA_ONNX_VERSION}${SHERPA_SUFFIX}" \
    -f "$CUDA_WHEEL_INDEX"

# --- Download models --------------------------------------------------------
echo ""
echo "Downloading models..."
python scripts/download_models.py

# --- Verify -----------------------------------------------------------------
echo ""
echo "Verifying CUDA support..."
python -c "
import sherpa_onnx
print(f'sherpa-onnx version: {sherpa_onnx.__version__}')
print('CUDA wheel installed successfully')
"

echo ""
echo "============================================================"
echo " Installation complete!"
echo ""
echo " Start server with CUDA:"
echo "   python -m live_transcript -c config_cuda.yaml"
echo ""
echo " Start server with CPU (fallback):"
echo "   python -m live_transcript"
echo "============================================================"
