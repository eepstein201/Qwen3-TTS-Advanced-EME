#!/bin/bash
# Qwen3-TTS Installation Script
# Cross-platform: macOS (MLX/torch) and Linux (torch/vLLM)

set -e

# =============================================================================
# Section 1: Configuration Constants
# =============================================================================

MLX_ENV_NAME="qwen3-tts-mlx"
TORCH_ENV_NAME="qwen3-tts"
PYTHON_VERSION="3.11"
USER_FILES_DIR="$HOME/Qwen3-TTS_UserFiles"
VOICE_PROMPTS_DIR="$USER_FILES_DIR/voice_prompts"
MIN_DISK_SPACE_GB=15

# Platform detection (set by detect_platform)
PLATFORM=""       # macos / linux
ARCH=""           # arm64 / x86_64 / aarch64
DISTRO_ID=""      # ubuntu, fedora, arch, etc.
DISTRO_VERSION="" # 22.04, 39, etc.
DISTRO_FAMILY=""  # debian / rhel / arch / suse / unknown

# Detected optimal settings (set by detect_optimal_settings)
RECOMMENDED_BACKEND=""
RECOMMENDED_SIZE=""
RECOMMENDED_QUANT=""
IS_INTEL=false
RAM_GB=0

# Linux GPU detection (set by detect_linux_gpu)
HAS_NVIDIA=false
NVIDIA_GPU_NAME=""
CUDA_VERSION=""
VRAM_GB=0
COMPUTE_CAP=""

# Linux install strategy (set by check_prerequisites)
LINUX_USE_CONDA=false

# User-selected settings
SELECTED_BACKEND=""
SELECTED_SIZE=""
SELECTED_QUANT=""

# =============================================================================
# Section 2: Color/Output Helpers
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
step() { echo -e "\n${CYAN}==> $1${NC}"; }

# =============================================================================
# Section 3: Platform Detection Dispatcher
# =============================================================================

detect_platform() {
    local os_name
    os_name=$(uname -s)
    ARCH=$(uname -m)

    case "$os_name" in
        Darwin)
            PLATFORM="macos"
            ;;
        Linux)
            PLATFORM="linux"
            # Detect distro from /etc/os-release
            if [[ -f /etc/os-release ]]; then
                # shellcheck source=/dev/null
                source /etc/os-release
                DISTRO_ID="${ID:-unknown}"
                DISTRO_VERSION="${VERSION_ID:-}"
                # Map to distro family
                case "$DISTRO_ID" in
                    ubuntu|debian|pop|linuxmint|elementary|zorin|kali)
                        DISTRO_FAMILY="debian" ;;
                    fedora|rhel|centos|rocky|alma|ol|amzn)
                        DISTRO_FAMILY="rhel" ;;
                    arch|manjaro|endeavouros|garuda)
                        DISTRO_FAMILY="arch" ;;
                    opensuse*|sles)
                        DISTRO_FAMILY="suse" ;;
                    *)
                        DISTRO_FAMILY="unknown" ;;
                esac
            else
                DISTRO_ID="unknown"
                DISTRO_FAMILY="unknown"
            fi
            ;;
        *)
            error "Unsupported operating system: $os_name"
            error "This script supports macOS and Linux only."
            exit 1
            ;;
    esac
}

# =============================================================================
# Section 4: Cross-Platform Utility Functions
# =============================================================================

detect_ram() {
    if [[ "$PLATFORM" == "macos" ]]; then
        RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
    elif [[ "$PLATFORM" == "linux" ]]; then
        RAM_GB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo 2>/dev/null)
    fi
    if [[ -z "$RAM_GB" ]] || [[ "$RAM_GB" -eq 0 ]]; then
        RAM_GB=16
        warn "Could not detect RAM — assuming 16GB"
    fi
}

detect_disk_space() {
    local available
    if [[ "$PLATFORM" == "macos" ]]; then
        available=$(df -g "$HOME" | awk 'NR==2 {print $4}')
    elif [[ "$PLATFORM" == "linux" ]]; then
        available=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    echo "${available:-0}"
}

source_conda() {
    # Check common conda install paths
    local conda_paths=(
        "$HOME/miniforge3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/mambaforge/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda/etc/profile.d/conda.sh"
    )

    for conda_sh in "${conda_paths[@]}"; do
        if [[ -f "$conda_sh" ]]; then
            # shellcheck source=/dev/null
            source "$conda_sh"
            return 0
        fi
    done

    # Fallback: use conda info --base if conda is on PATH
    if command -v conda &>/dev/null; then
        local conda_base
        conda_base=$(conda info --base 2>/dev/null)
        if [[ -n "$conda_base" ]] && [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "$conda_base/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    # Last resort: search HOME
    local found
    found=$(find "$HOME" -maxdepth 4 -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        # shellcheck source=/dev/null
        source "$found"
        return 0
    fi

    error "Could not find conda.sh. Please ensure conda is properly installed."
    return 1
}

# =============================================================================
# Section 5: Linux GPU Detection
# =============================================================================

detect_linux_gpu() {
    [[ "$PLATFORM" != "linux" ]] && return

    step "Detecting GPU..."

    # 1. Check for NVIDIA GPU via PCI vendor ID
    if command -v lspci &>/dev/null; then
        if lspci -d '10de:' 2>/dev/null | grep -qi 'vga\|3d\|display'; then
            HAS_NVIDIA=true
            NVIDIA_GPU_NAME=$(lspci -d '10de:' 2>/dev/null | grep -i 'vga\|3d\|display' | head -1 | sed 's/.*: //')
        fi
    fi

    # Fallback: check /dev/nvidia0
    if [[ "$HAS_NVIDIA" == false ]] && [[ -e /dev/nvidia0 ]]; then
        HAS_NVIDIA=true
        NVIDIA_GPU_NAME="NVIDIA GPU (detected via /dev/nvidia0)"
    fi

    if [[ "$HAS_NVIDIA" == false ]]; then
        info "No NVIDIA GPU detected — will install CPU-only PyTorch"
        return
    fi

    success "NVIDIA GPU detected: $NVIDIA_GPU_NAME"

    # 2. Detect CUDA version (5-method cascade)
    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -o 'release [0-9]\+\.[0-9]\+' | sed 's/release //' | head -1)
    fi

    if [[ -z "$CUDA_VERSION" ]] && command -v nvidia-smi &>/dev/null; then
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]\+\.[0-9]\+' | sed 's/CUDA Version: //' | head -1)
    fi

    if [[ -z "$CUDA_VERSION" ]] && [[ -f /usr/local/cuda/version.json ]]; then
        CUDA_VERSION=$(python3 -c "import json; print('.'.join(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'].split('.')[:2]))" 2>/dev/null)
    fi

    if [[ -z "$CUDA_VERSION" ]]; then
        # Try package manager
        if command -v dpkg &>/dev/null; then
            CUDA_VERSION=$(dpkg -l 2>/dev/null | grep 'cuda-toolkit' | awk '{print $3}' | grep -o '^[0-9]\+\.[0-9]\+' | head -1)
        elif command -v rpm &>/dev/null; then
            CUDA_VERSION=$(rpm -qa 2>/dev/null | grep 'cuda-toolkit' | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        fi
    fi

    if [[ -z "$CUDA_VERSION" ]] && command -v ldconfig &>/dev/null; then
        CUDA_VERSION=$(ldconfig -p 2>/dev/null | grep libcudart | grep -o '[0-9]\+\.[0-9]\+' | sort -V | tail -1)
    fi

    if [[ -n "$CUDA_VERSION" ]]; then
        success "CUDA version: $CUDA_VERSION"
    else
        warn "NVIDIA GPU detected but no CUDA toolkit found."
        warn "Install CUDA from: https://developer.nvidia.com/cuda-downloads"
        warn "Continuing with CPU-only PyTorch."
    fi

    # 3. Detect VRAM
    if command -v nvidia-smi &>/dev/null; then
        VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}')
        if [[ -n "$VRAM_GB" ]] && [[ "$VRAM_GB" -gt 0 ]]; then
            info "VRAM: ${VRAM_GB}GB"
        fi

        # 4. Detect compute capability
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
        if [[ -n "$COMPUTE_CAP" ]]; then
            info "Compute capability: $COMPUTE_CAP"
        fi
    fi
}

# =============================================================================
# Section 6: Linux System Dependencies
# =============================================================================

install_linux_system_deps() {
    [[ "$PLATFORM" != "linux" ]] && return

    step "Checking system dependencies..."

    local missing=()

    # Check ffmpeg
    if ! command -v ffmpeg &>/dev/null; then
        missing+=("ffmpeg")
    else
        info "ffmpeg: installed"
    fi

    # Check libsndfile
    if ! ldconfig -p 2>/dev/null | grep -q libsndfile; then
        missing+=("libsndfile")
    else
        info "libsndfile: installed"
    fi

    # Check rubberband
    if ! ldconfig -p 2>/dev/null | grep -q librubberband; then
        missing+=("rubberband")
    else
        info "rubberband: installed"
    fi

    if [[ ${#missing[@]} -eq 0 ]]; then
        success "All system dependencies installed"
        return
    fi

    warn "Missing system dependencies: ${missing[*]}"

    # Map to distro-specific package names
    local pkgs=()
    for dep in "${missing[@]}"; do
        case "$DISTRO_FAMILY" in
            debian)
                case "$dep" in
                    ffmpeg) pkgs+=("ffmpeg") ;;
                    libsndfile) pkgs+=("libsndfile1-dev") ;;
                    rubberband) pkgs+=("librubberband-dev") ;;
                esac ;;
            rhel)
                case "$dep" in
                    ffmpeg) pkgs+=("ffmpeg") ;;
                    libsndfile) pkgs+=("libsndfile-devel") ;;
                    rubberband) pkgs+=("rubberband-devel") ;;
                esac ;;
            arch)
                case "$dep" in
                    ffmpeg) pkgs+=("ffmpeg") ;;
                    libsndfile) pkgs+=("libsndfile") ;;
                    rubberband) pkgs+=("rubberband") ;;
                esac ;;
            suse)
                case "$dep" in
                    ffmpeg) pkgs+=("ffmpeg") ;;
                    libsndfile) pkgs+=("libsndfile-devel") ;;
                    rubberband) pkgs+=("rubberband-devel") ;;
                esac ;;
            *)
                warn "Unknown distro ($DISTRO_ID). Please install manually: ${missing[*]}"
                warn "  ffmpeg: https://ffmpeg.org/download.html"
                warn "  libsndfile: https://github.com/libsndfile/libsndfile"
                warn "  rubberband: https://breakfastquay.com/rubberband/"
                return
                ;;
        esac
    done

    echo ""
    case "$DISTRO_FAMILY" in
        debian) echo "Install command: sudo apt-get install -y ${pkgs[*]}" ;;
        rhel)   echo "Install command: sudo dnf install -y ${pkgs[*]}" ;;
        arch)   echo "Install command: sudo pacman -S --noconfirm ${pkgs[*]}" ;;
        suse)   echo "Install command: sudo zypper install -y ${pkgs[*]}" ;;
    esac

    echo ""
    read -p "Install missing dependencies now? (requires sudo) [Y/n]: " INSTALL_DEPS
    INSTALL_DEPS=${INSTALL_DEPS:-Y}

    if [[ "$INSTALL_DEPS" =~ ^[Yy]$ ]]; then
        case "$DISTRO_FAMILY" in
            debian) sudo apt-get update && sudo apt-get install -y "${pkgs[@]}" ;;
            rhel)   sudo dnf install -y "${pkgs[@]}" ;;
            arch)   sudo pacman -S --noconfirm "${pkgs[@]}" ;;
            suse)   sudo zypper install -y "${pkgs[@]}" ;;
        esac

        if [[ $? -eq 0 ]]; then
            success "System dependencies installed"
        else
            warn "Some packages may have failed to install. Continuing anyway."
        fi
    else
        warn "Skipping system dependency install. Some features may not work."
        if [[ " ${missing[*]} " == *" rubberband "* ]]; then
            warn "  Without rubberband, speed/pitch processing falls back to librosa"
        fi
    fi
}

# =============================================================================
# Section 7: PyTorch CUDA Index URL
# =============================================================================

get_torch_index_url() {
    # No GPU or no CUDA → CPU
    if [[ "$HAS_NVIDIA" == false ]] || [[ -z "$CUDA_VERSION" ]]; then
        echo "https://download.pytorch.org/whl/cpu"
        return
    fi

    # Parse major.minor
    local major minor
    major=$(echo "$CUDA_VERSION" | cut -d. -f1)
    minor=$(echo "$CUDA_VERSION" | cut -d. -f2)
    local cuda_num=$((major * 10 + minor))

    if [[ $cuda_num -ge 126 ]]; then
        echo "https://download.pytorch.org/whl/cu126"
    elif [[ $cuda_num -ge 124 ]]; then
        echo "https://download.pytorch.org/whl/cu124"
    elif [[ $cuda_num -ge 121 ]]; then
        echo "https://download.pytorch.org/whl/cu121"
    elif [[ $cuda_num -ge 118 ]]; then
        echo "https://download.pytorch.org/whl/cu118"
    else
        warn "CUDA $CUDA_VERSION is too old (< 11.8). Installing CPU-only PyTorch."
        echo "https://download.pytorch.org/whl/cpu"
    fi
}

_pip_install_linux() {
    local index_url
    index_url=$(get_torch_index_url)

    info "PyTorch index URL: $index_url"
    pip install torch torchvision torchaudio --index-url "$index_url"

    local extras="torch,server,audio,rich,ui"
    if [[ "$HAS_NVIDIA" == true ]] && [[ -n "$CUDA_VERSION" ]]; then
        extras="torch,cuda,server,audio,rich,ui"
    fi
    pip install -e "$USER_FILES_DIR/[$extras]"
}

# =============================================================================
# Section 8: Linux venv Fallback
# =============================================================================

create_linux_venv() {
    step "Setting up Python virtual environment..."

    # Find Python 3.10+
    local python_cmd=""
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local ver_major ver_minor
            ver_major=$(echo "$ver" | cut -d. -f1)
            ver_minor=$(echo "$ver" | cut -d. -f2)
            if [[ "$ver_major" -ge 3 ]] && [[ "$ver_minor" -ge 10 ]]; then
                python_cmd="$candidate"
                info "Found $candidate (Python $ver)"
                break
            fi
        fi
    done

    if [[ -z "$python_cmd" ]]; then
        error "Python 3.10+ is required but not found."
        error "Install Python 3.10+ for your system:"
        case "$DISTRO_FAMILY" in
            debian) error "  sudo apt-get install python3.11 python3.11-venv" ;;
            rhel)   error "  sudo dnf install python3.11" ;;
            arch)   error "  sudo pacman -S python" ;;
            suse)   error "  sudo zypper install python311" ;;
            *)      error "  See https://www.python.org/downloads/" ;;
        esac
        exit 1
    fi

    local venv_dir="$USER_FILES_DIR/.venv"

    if [[ -d "$venv_dir" ]]; then
        warn "Virtual environment already exists at $venv_dir"
        read -p "Recreate environment? This will delete the existing one. [y/N]: " RECREATE
        if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
            info "Removing existing virtual environment..."
            rm -rf "$venv_dir"
        else
            info "Keeping existing environment. Updating packages..."
            # shellcheck source=/dev/null
            source "$venv_dir/bin/activate"
            _pip_install_linux
            deactivate 2>/dev/null || true
            success "Packages updated!"
            return
        fi
    fi

    info "Creating virtual environment with $python_cmd..."
    "$python_cmd" -m venv "$venv_dir"

    # shellcheck source=/dev/null
    source "$venv_dir/bin/activate"
    pip install --upgrade pip

    _pip_install_linux

    deactivate 2>/dev/null || true

    success "Virtual environment created at $venv_dir"
}

# =============================================================================
# Section 9: Hardware Detection
# =============================================================================

detect_optimal_settings() {
    step "Detecting hardware..."

    detect_ram

    if [[ "$PLATFORM" == "macos" ]]; then
        # macOS: check Apple Silicon vs Intel
        if [[ "$ARCH" == "arm64" ]]; then
            RECOMMENDED_BACKEND="mlx"
            IS_INTEL=false
            success "Apple Silicon detected (arm64) — MLX backend available"
        else
            RECOMMENDED_BACKEND="torch"
            IS_INTEL=true
            warn "Intel Mac detected ($ARCH) — MLX not available, using PyTorch backend"
        fi

        # Size recommendation based on RAM
        if [[ "$RAM_GB" -lt 16 ]]; then
            RECOMMENDED_SIZE="0.6B"
            RECOMMENDED_QUANT="4bit"
            info "RAM: ${RAM_GB}GB — recommending smaller models (0.6B, 4bit)"
        else
            RECOMMENDED_SIZE="1.7B"
            RECOMMENDED_QUANT="8bit"
            info "RAM: ${RAM_GB}GB — recommending standard models (1.7B, 8bit)"
        fi

    elif [[ "$PLATFORM" == "linux" ]]; then
        IS_INTEL=false
        RECOMMENDED_BACKEND="torch"
        RECOMMENDED_QUANT="none"

        detect_linux_gpu

        # Size recommendation: use VRAM if available, else RAM
        local ref_mem=$RAM_GB
        if [[ "$HAS_NVIDIA" == true ]] && [[ "$VRAM_GB" -gt 0 ]]; then
            ref_mem=$VRAM_GB
        fi

        if [[ "$ref_mem" -lt 8 ]]; then
            RECOMMENDED_SIZE="0.6B"
            info "Memory: ${ref_mem}GB — recommending smaller model (0.6B)"
        else
            RECOMMENDED_SIZE="1.7B"
            info "Memory: ${ref_mem}GB — recommending standard model (1.7B)"
        fi

        if [[ "$HAS_NVIDIA" == true ]]; then
            success "Linux with NVIDIA GPU — PyTorch CUDA backend"
        else
            info "Linux (CPU only) — PyTorch CPU backend"
        fi
    fi

    # Set defaults
    SELECTED_BACKEND="$RECOMMENDED_BACKEND"
    SELECTED_SIZE="$RECOMMENDED_SIZE"
    SELECTED_QUANT="$RECOMMENDED_QUANT"
}

# =============================================================================
# Section 10: Interactive Configuration Wizard
# =============================================================================

run_config_wizard() {
    step "Configuration Wizard"

    echo ""
    echo -e "${BOLD}Hardware Detection${NC}"
    echo "=================="
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  Platform: macOS"
        echo "  Chip: $(uname -m)"
        echo "  RAM:  ${RAM_GB}GB"
    elif [[ "$PLATFORM" == "linux" ]]; then
        echo "  Platform: Linux ($DISTRO_ID${DISTRO_VERSION:+ $DISTRO_VERSION})"
        echo "  Arch: $ARCH"
        echo "  RAM:  ${RAM_GB}GB"
        if [[ "$HAS_NVIDIA" == true ]]; then
            echo "  GPU:  $NVIDIA_GPU_NAME"
            [[ -n "$CUDA_VERSION" ]] && echo "  CUDA: $CUDA_VERSION"
            [[ "$VRAM_GB" -gt 0 ]] && echo "  VRAM: ${VRAM_GB}GB"
        else
            echo "  GPU:  None detected (CPU only)"
        fi
    fi
    echo ""

    echo -e "${BOLD}Recommended Configuration${NC}"
    echo "========================="
    if [[ "$PLATFORM" == "macos" ]]; then
        if [[ "$IS_INTEL" == true ]]; then
            echo "  Backend: torch (Intel Mac — MLX not available)"
        else
            echo "  Backend: $RECOMMENDED_BACKEND (native Apple Silicon)"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        if [[ "$HAS_NVIDIA" == true ]] && [[ -n "$CUDA_VERSION" ]]; then
            echo "  Backend: torch (CUDA)"
        else
            echo "  Backend: torch (CPU)"
        fi
    fi
    echo "  Model size: $RECOMMENDED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo "  Quantization: $RECOMMENDED_QUANT (MLX only)"
    fi
    echo ""

    read -p "Apply recommended settings? [Y/n/c for customize]: " CHOICE
    CHOICE=${CHOICE:-Y}

    if [[ "$CHOICE" =~ ^[Cc]$ ]]; then
        customize_settings
    elif [[ ! "$CHOICE" =~ ^[Yy]$ ]]; then
        echo "Using recommended settings."
    fi

    echo ""
    echo -e "${BOLD}Selected Configuration${NC}"
    echo "======================"
    echo "  Backend: $SELECTED_BACKEND"
    echo "  Model size: $SELECTED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo "  Quantization: $SELECTED_QUANT"
    fi
    echo ""
}

customize_settings() {
    echo ""

    if [[ "$PLATFORM" == "macos" ]]; then
        # Backend selection (only for Apple Silicon)
        if [[ "$IS_INTEL" == false ]]; then
            echo -e "${BOLD}Backend${NC}"
            echo "  1. mlx — Apple Silicon native, lower thermals (recommended)"
            echo "  2. torch — PyTorch/MPS, if you prefer or need torch compatibility"
            read -p "Select [1]: " BACKEND_CHOICE
            BACKEND_CHOICE=${BACKEND_CHOICE:-1}

            if [[ "$BACKEND_CHOICE" == "2" ]]; then
                SELECTED_BACKEND="torch"
            else
                SELECTED_BACKEND="mlx"
            fi
        else
            echo -e "${YELLOW}Backend: torch (Intel Mac — no other options)${NC}"
            SELECTED_BACKEND="torch"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        if [[ "$HAS_NVIDIA" == true ]] && [[ -n "$CUDA_VERSION" ]]; then
            echo -e "${BOLD}Backend${NC}"
            echo "  1. torch — PyTorch with CUDA (recommended)"
            echo "  2. vllm  — vLLM optimized serving (advanced, requires more VRAM)"
            read -p "Select [1]: " BACKEND_CHOICE
            BACKEND_CHOICE=${BACKEND_CHOICE:-1}

            if [[ "$BACKEND_CHOICE" == "2" ]]; then
                SELECTED_BACKEND="vllm"
            else
                SELECTED_BACKEND="torch"
            fi
        else
            echo -e "${YELLOW}Backend: torch (CPU only — no NVIDIA GPU detected)${NC}"
            SELECTED_BACKEND="torch"
        fi
    fi

    echo ""

    # Model size selection
    echo -e "${BOLD}Model Size${NC}"
    echo "  1. 1.7B — Higher quality, ~3.5GB per model"
    echo "  2. 0.6B — Faster (~40%), lower memory, good for <16GB RAM"
    read -p "Select [1]: " SIZE_CHOICE
    SIZE_CHOICE=${SIZE_CHOICE:-1}

    if [[ "$SIZE_CHOICE" == "2" ]]; then
        SELECTED_SIZE="0.6B"
    else
        SELECTED_SIZE="1.7B"
    fi

    # MLX Quantization (only if MLX backend selected)
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo ""
        echo -e "${BOLD}MLX Quantization${NC}"
        echo "  1. 8bit — Good balance of quality and size (recommended)"
        echo "  2. 4bit — Smallest, fastest, slightly lower quality"
        echo "  3. 5bit — Slightly higher quality than 4bit"
        echo "  4. 6bit — Closer to 8bit quality, smaller than 8bit"
        echo "  5. bf16 — Full precision, highest quality, largest size"
        read -p "Select [1]: " QUANT_CHOICE
        QUANT_CHOICE=${QUANT_CHOICE:-1}

        case "$QUANT_CHOICE" in
            2) SELECTED_QUANT="4bit" ;;
            3) SELECTED_QUANT="5bit" ;;
            4) SELECTED_QUANT="6bit" ;;
            5) SELECTED_QUANT="bf16" ;;
            *) SELECTED_QUANT="8bit" ;;
        esac
    fi
}

# =============================================================================
# Section 11: Prerequisite Checks
# =============================================================================

check_prerequisites() {
    step "Checking prerequisites..."

    if [[ "$PLATFORM" == "macos" ]]; then
        info "macOS detected"

        # Check for conda
        if ! command -v conda &>/dev/null; then
            error "Conda is not installed or not in PATH."
            echo ""
            echo "Please install Miniforge first:"
            echo "  brew install --cask miniforge"
            echo "  conda init zsh  # or bash"
            echo ""
            echo "Then run this script again."
            exit 1
        fi
        success "Conda is installed"

        # Install rubberband via Homebrew
        if command -v brew &>/dev/null; then
            if ! brew list rubberband &>/dev/null 2>&1; then
                info "Installing rubberband (for high-quality speed/pitch adjustment)..."
                brew install rubberband
                success "rubberband installed"
            else
                info "rubberband already installed"
            fi
        else
            warn "Homebrew not found — install rubberband manually for best audio quality:"
            warn "  brew install rubberband"
            warn "  (falling back to librosa if not installed)"
        fi

    elif [[ "$PLATFORM" == "linux" ]]; then
        info "Linux detected ($DISTRO_ID${DISTRO_VERSION:+ $DISTRO_VERSION}, $DISTRO_FAMILY family)"

        # Install system dependencies (ffmpeg, libsndfile, rubberband)
        install_linux_system_deps

        # Check for conda or python
        if command -v conda &>/dev/null; then
            LINUX_USE_CONDA=true
            success "Conda is installed — will use conda environment"
        else
            LINUX_USE_CONDA=false
            info "Conda not found — will use Python venv"

            # Verify python3 exists
            if ! command -v python3 &>/dev/null; then
                error "Python 3 is not installed."
                case "$DISTRO_FAMILY" in
                    debian) error "  Install with: sudo apt-get install python3 python3-venv" ;;
                    rhel)   error "  Install with: sudo dnf install python3" ;;
                    arch)   error "  Install with: sudo pacman -S python" ;;
                    suse)   error "  Install with: sudo zypper install python3" ;;
                    *)      error "  See https://www.python.org/downloads/" ;;
                esac
                exit 1
            fi
        fi
    fi

    # Check disk space (cross-platform)
    local available_gb
    available_gb=$(detect_disk_space)
    if [[ "$available_gb" -lt "$MIN_DISK_SPACE_GB" ]]; then
        warn "Low disk space: ${available_gb}GB available (recommended: ${MIN_DISK_SPACE_GB}GB+)"
        warn "Models require ~10GB of disk space."
        read -p "Continue anyway? [y/N]: " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    else
        info "Disk space: ${available_gb}GB available"
    fi

    success "All prerequisites met!"
}

# =============================================================================
# Section 12: Create MLX Conda Environment (macOS Apple Silicon only)
# =============================================================================

create_mlx_env() {
    # MLX is macOS Apple Silicon only
    if [[ "$PLATFORM" != "macos" ]]; then
        return
    fi

    if [[ "$IS_INTEL" == true ]]; then
        info "Skipping MLX environment (Intel Mac)"
        return
    fi

    if [[ "$SELECTED_BACKEND" != "mlx" ]]; then
        info "MLX environment not needed (torch backend selected)"
        return
    fi

    step "Setting up MLX environment: $MLX_ENV_NAME..."

    source_conda

    # Check if environment exists
    if conda env list | grep -q "^${MLX_ENV_NAME} "; then
        warn "Environment '$MLX_ENV_NAME' already exists."
        read -p "Recreate environment? This will delete the existing one. [y/N]: " RECREATE
        if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
            info "Removing existing environment..."
            conda env remove -n "$MLX_ENV_NAME" -y
        else
            info "Keeping existing environment. Updating packages..."
            conda activate "$MLX_ENV_NAME"
            pip install --upgrade -e "$USER_FILES_DIR/[mlx,server,audio,rich,ui]"
            conda deactivate 2>/dev/null || true
            success "Packages updated!"
            return
        fi
    fi

    info "Creating new conda environment with Python $PYTHON_VERSION..."
    conda create -n "$MLX_ENV_NAME" python="$PYTHON_VERSION" -y

    info "Activating environment..."
    conda activate "$MLX_ENV_NAME"

    info "Installing MLX dependencies..."
    pip install -e "$USER_FILES_DIR/[mlx,server,audio,rich,ui]"

    conda deactivate 2>/dev/null || true

    success "MLX environment created and packages installed!"
}

# =============================================================================
# Section 13: Create Torch Environment (macOS + Linux conda)
# =============================================================================

create_torch_env() {
    if [[ "$PLATFORM" == "macos" ]]; then
        _create_torch_env_macos
    elif [[ "$PLATFORM" == "linux" ]]; then
        if [[ "$LINUX_USE_CONDA" == true ]]; then
            _create_torch_env_linux_conda
        else
            create_linux_venv
        fi
    fi
}

_create_torch_env_macos() {
    # For Intel: always required
    # For Apple Silicon: optional (if user wants torch fallback or selected torch)
    if [[ "$IS_INTEL" == true ]]; then
        step "Setting up PyTorch environment: $TORCH_ENV_NAME (required for Intel Mac)..."
    elif [[ "$SELECTED_BACKEND" == "torch" ]]; then
        step "Setting up PyTorch environment: $TORCH_ENV_NAME..."
    else
        # Apple Silicon with MLX — offer torch as optional fallback
        echo ""
        read -p "Install PyTorch fallback environment? (for sharing with Intel Mac users) [y/N]: " INSTALL_TORCH
        INSTALL_TORCH=${INSTALL_TORCH:-N}

        if [[ ! "$INSTALL_TORCH" =~ ^[Yy]$ ]]; then
            info "Skipping PyTorch environment."
            return
        fi
        step "Setting up PyTorch fallback environment: $TORCH_ENV_NAME..."
    fi

    source_conda

    # Check if environment exists
    if conda env list | grep -q "^${TORCH_ENV_NAME} "; then
        warn "Environment '$TORCH_ENV_NAME' already exists."
        read -p "Recreate environment? This will delete the existing one. [y/N]: " RECREATE
        if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
            info "Removing existing environment..."
            conda env remove -n "$TORCH_ENV_NAME" -y
        else
            info "Keeping existing environment. Updating packages..."
            conda activate "$TORCH_ENV_NAME"
            pip install --upgrade -e "$USER_FILES_DIR/[torch,server,audio,ui,rich]"
            conda deactivate 2>/dev/null || true
            success "Packages updated!"
            return
        fi
    fi

    info "Creating new conda environment with Python $PYTHON_VERSION..."
    conda create -n "$TORCH_ENV_NAME" python="$PYTHON_VERSION" -y

    info "Activating environment..."
    conda activate "$TORCH_ENV_NAME"

    info "Installing PyTorch with MPS support..."
    pip install torch torchvision torchaudio

    info "Installing Qwen3-TTS and dependencies..."
    pip install -e "$USER_FILES_DIR/[torch,server,audio,ui,rich]"

    conda deactivate 2>/dev/null || true

    success "PyTorch environment created and packages installed!"
}

_create_torch_env_linux_conda() {
    step "Setting up PyTorch environment: $TORCH_ENV_NAME..."

    source_conda

    # Check if environment exists
    if conda env list | grep -q "^${TORCH_ENV_NAME} "; then
        warn "Environment '$TORCH_ENV_NAME' already exists."
        read -p "Recreate environment? This will delete the existing one. [y/N]: " RECREATE
        if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
            info "Removing existing environment..."
            conda env remove -n "$TORCH_ENV_NAME" -y
        else
            info "Keeping existing environment. Updating packages..."
            conda activate "$TORCH_ENV_NAME"
            _pip_install_linux
            conda deactivate 2>/dev/null || true
            success "Packages updated!"
            return
        fi
    fi

    info "Creating new conda environment with Python $PYTHON_VERSION..."
    conda create -n "$TORCH_ENV_NAME" python="$PYTHON_VERSION" -y

    info "Activating environment..."
    conda activate "$TORCH_ENV_NAME"

    _pip_install_linux

    conda deactivate 2>/dev/null || true

    success "PyTorch environment created and packages installed!"
}

# =============================================================================
# Section 14: Create Directories
# =============================================================================

create_directories() {
    step "Creating directories..."

    mkdir -p "$USER_FILES_DIR"
    mkdir -p "$VOICE_PROMPTS_DIR"

    success "Directories created:"
    info "  $USER_FILES_DIR"
    info "  $VOICE_PROMPTS_DIR"
}

# =============================================================================
# Section 15: Create Config File
# =============================================================================

# Helper: determine dtype for Linux based on compute capability
get_linux_dtype() {
    if [[ "$HAS_NVIDIA" == false ]] || [[ -z "$COMPUTE_CAP" ]]; then
        echo "float32"
        return
    fi
    local major
    if [[ "$COMPUTE_CAP" == *"."* ]]; then
        major=$(echo "$COMPUTE_CAP" | cut -d. -f1)
    else
        major=${COMPUTE_CAP:0:1}  # First digit for formats like "89"
    fi
    if [[ "$major" -ge 8 ]]; then
        echo "bfloat16"  # Ampere+ (A100, L4, RTX 30xx+)
    else
        echo "float16"   # Turing (T4, RTX 20xx)
    fi
}

# Helper: determine torch quantization for Linux based on compute capability
get_linux_torch_quant() {
    if [[ "$HAS_NVIDIA" == false ]] || [[ -z "$COMPUTE_CAP" ]]; then
        echo "none"
        return
    fi
    local major
    if [[ "$COMPUTE_CAP" == *"."* ]]; then
        major=$(echo "$COMPUTE_CAP" | cut -d. -f1)
    else
        major=${COMPUTE_CAP:0:1}
    fi
    if [[ "$major" -ge 8 ]]; then
        echo "none"  # Ampere+ has enough VRAM
    else
        echo "8bit"  # Turing benefits from quantization
    fi
}

create_config() {
    step "Creating configuration file..."

    CONFIG_FILE="$USER_FILES_DIR/config.json"

    if [[ -f "$CONFIG_FILE" ]] && [[ "$RECONFIGURE_ONLY" != true ]]; then
        warn "config.json already exists."
        read -p "Merge new settings into existing config? [Y/n]: " MERGE
        if [[ "$MERGE" =~ ^[Nn]$ ]]; then
            info "Keeping existing config.json (skipping update)."
            return
        fi
        # Merge new fields into existing config using Python
        info "Merging new settings into existing config.json..."
        python3 << PYTHON_EOF
import json

config_file = "$CONFIG_FILE"
selected_backend = "$SELECTED_BACKEND"

with open(config_file, 'r') as f:
    config = json.load(f)

# Add advanced section fields if missing
if "advanced" not in config:
    config["advanced"] = {}
if "torch_quantization" not in config.get("advanced", {}):
    config["advanced"]["torch_quantization"] = "none"
if "audio_loader" not in config.get("advanced", {}):
    config["advanced"]["audio_loader"] = "torchaudio"
if "vllm_gpu_memory_utilization" not in config.get("advanced", {}):
    config["advanced"]["vllm_gpu_memory_utilization"] = 0.7
if "vllm_port" not in config.get("advanced", {}):
    config["advanced"]["vllm_port"] = None

# Add cache section if missing
if "cache" not in config:
    config["cache"] = {}
if "voice_prompt_max" not in config.get("cache", {}):
    config["cache"]["voice_prompt_max"] = 10
if "generation_max" not in config.get("cache", {}):
    config["cache"]["generation_max"] = 5
if "eta_ttl_seconds" not in config.get("cache", {}):
    config["cache"]["eta_ttl_seconds"] = 30

# Add generation section fields if missing
if "generation" not in config:
    config["generation"] = {}
if "max_chunk_tokens" not in config.get("generation", {}):
    config["generation"]["max_chunk_tokens"] = 200
if "max_new_tokens" not in config.get("generation", {}):
    config["generation"]["max_new_tokens"] = 2048
if "compile_model" not in config.get("generation", {}):
    config["generation"]["compile_model"] = True

# Add prosody_presets section if missing
if "prosody_presets" not in config:
    config["prosody_presets"] = {
        "excited": "Speak with excitement and high energy",
        "calm": "Speak in a calm, soothing, relaxed manner",
        "whisper": "Speak in a soft whisper",
        "authoritative": "Speak in a confident, authoritative tone",
        "slow": "Speak slowly and deliberately with clear enunciation",
        "fast": "Speak quickly with urgency",
        "dramatic": "Speak with dramatic flair and emotional intensity",
        "conversational": "Speak in a casual, natural conversational style"
    }

# Add prompt_enhancer section if missing
if "prompt_enhancer" not in config:
    config["prompt_enhancer"] = {
        "enabled": False,
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-haiku-4-5-20251001"
    }

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
print("Merged new settings into config.json")
PYTHON_EOF
        success "Config merged with new settings!"
        return
    fi

    # Determine platform-specific defaults
    local config_dtype="bfloat16"
    local config_torch_quant="none"
    if [[ "$PLATFORM" == "linux" ]]; then
        config_dtype=$(get_linux_dtype)
        config_torch_quant=$(get_linux_torch_quant)
    fi

    # Create new config file
    cat > "$CONFIG_FILE" << EOF
{
  "default_voice_description": "A calm, friendly male voice with clear articulation and moderate pace.",
  "default_clone_prompt": "default_clone.pt",
  "default_speaker": "ryan",
  "output_directory": "~/Downloads",
  "language": "English",
  "server": {
    "host": "127.0.0.1",
    "port": 5123,
    "auto_shutdown_minutes": 0
  },
  "models": {
    "clone": {
      "load_at_startup": true,
      "description": "Voice cloning from audio samples (clone mode, -p prompt.pt)"
    },
    "design": {
      "load_at_startup": false,
      "description": "Generate voice from text description (design mode, -d 'description')"
    },
    "custom": {
      "load_at_startup": false,
      "description": "9 premium pre-trained speakers (custom mode, -s ryan)"
    }
  },
  "security": {
    "max_text_length": 10000,
    "max_batch_size": 20
  },
  "advanced": {
    "dtype": "$config_dtype",
    "backend": "$SELECTED_BACKEND",
    "mlx_quantization": "$SELECTED_QUANT",
    "torch_quantization": "$config_torch_quant",
    "model_size": "$SELECTED_SIZE",
    "audio_loader": "torchaudio",
    "vllm_gpu_memory_utilization": 0.7,
    "vllm_port": null
  },
  "cache": {
    "voice_prompt_max": 10,
    "generation_max": 5,
    "eta_ttl_seconds": 30
  },
  "generation": {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "seed": null,
    "max_chunk_chars": 500,
    "max_chunk_tokens": 200,
    "max_new_tokens": 2048,
    "compile_model": true
  },
  "prosody_presets": {
    "excited": "Speak with excitement and high energy",
    "calm": "Speak in a calm, soothing, relaxed manner",
    "whisper": "Speak in a soft whisper",
    "authoritative": "Speak in a confident, authoritative tone",
    "slow": "Speak slowly and deliberately with clear enunciation",
    "fast": "Speak quickly with urgency",
    "dramatic": "Speak with dramatic flair and emotional intensity",
    "conversational": "Speak in a casual, natural conversational style"
  },
  "prompt_enhancer": {
    "enabled": false,
    "provider": "anthropic",
    "api_key_env": "ANTHROPIC_API_KEY",
    "model": "claude-haiku-4-5-20251001"
  },
  "presets": {
    "consistent": {
      "temperature": 0.5,
      "top_k": 30,
      "seed": 42
    },
    "creative": {
      "temperature": 0.9,
      "top_p": 0.98
    }
  },
  "ui": {
    "port": 7860
  },
  "aliases": {
    "default": {
      "prompt": "default_clone.pt",
      "preset": "consistent"
    }
  }
}
EOF

    success "Created config.json with selected settings:"
    info "  Backend: $SELECTED_BACKEND"
    info "  Model size: $SELECTED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        info "  Quantization: $SELECTED_QUANT"
    fi
    if [[ "$PLATFORM" == "linux" ]]; then
        info "  dtype: $config_dtype"
        [[ "$config_torch_quant" != "none" ]] && info "  Torch quantization: $config_torch_quant"
    fi
}

# =============================================================================
# Section 16: Update PATH
# =============================================================================

update_path() {
    step "CLI access setup..."

    if [[ "$PLATFORM" == "macos" ]]; then
        # Determine which environment was installed
        if [[ "$IS_INTEL" == true ]]; then
            ACTIVE_ENV="$TORCH_ENV_NAME"
        else
            ACTIVE_ENV="$MLX_ENV_NAME"
        fi

        echo ""
        echo "The 'tts' command is installed via pip entry points."
        echo "To use it, simply activate your conda environment:"
        echo ""
        echo -e "  ${CYAN}conda activate $ACTIVE_ENV${NC}"
        echo -e "  ${CYAN}tts --help${NC}"
        echo ""
        success "CLI ready! Use 'conda activate $ACTIVE_ENV' to access the 'tts' command."

    elif [[ "$PLATFORM" == "linux" ]]; then
        echo ""
        echo "The 'tts' command is installed via pip entry points."
        echo ""
        if [[ "$LINUX_USE_CONDA" == true ]]; then
            echo "To use it, activate your conda environment:"
            echo ""
            echo -e "  ${CYAN}conda activate $TORCH_ENV_NAME${NC}"
            echo -e "  ${CYAN}tts --help${NC}"
            echo ""
            success "CLI ready! Use 'conda activate $TORCH_ENV_NAME' to access the 'tts' command."
        else
            echo "To use it, activate your virtual environment:"
            echo ""
            echo -e "  ${CYAN}source $USER_FILES_DIR/.venv/bin/activate${NC}"
            echo -e "  ${CYAN}tts --help${NC}"
            echo ""
            echo "Tip: Add an alias to your shell profile:"
            echo -e "  ${CYAN}echo 'alias tts-env=\"source $USER_FILES_DIR/.venv/bin/activate\"' >> ~/.bashrc${NC}"
            echo ""
            success "CLI ready! Use 'source .venv/bin/activate' to access the 'tts' command."
        fi
    fi
}

# =============================================================================
# Section 17: Download Models
# =============================================================================

download_models() {
    step "Model pre-download (optional)..."

    echo ""
    echo "Models will be downloaded on first use (~3-10GB depending on settings)."
    echo "You can pre-download them now or skip this step."
    echo ""
    read -p "Download models now? [y/N]: " DOWNLOAD
    DOWNLOAD=${DOWNLOAD:-N}

    if [[ ! "$DOWNLOAD" =~ ^[Yy]$ ]]; then
        info "Skipping model download. Models will be downloaded on first use."
        return
    fi

    source_conda 2>/dev/null || true

    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        download_mlx_models
    else
        download_torch_models
    fi
}

download_mlx_models() {
    info "Downloading MLX models ($SELECTED_SIZE, $SELECTED_QUANT)..."
    conda activate "$MLX_ENV_NAME"

    python3 << PYEOF
import sys
try:
    from huggingface_hub import snapshot_download

    size = "$SELECTED_SIZE"
    quant = "$SELECTED_QUANT"

    models = [
        f"mlx-community/Qwen3-TTS-12Hz-{size}-Base-{quant}",
        f"mlx-community/Qwen3-TTS-12Hz-{size}-VoiceDesign-{quant}",
        f"mlx-community/Qwen3-TTS-12Hz-{size}-CustomVoice-{quant}"
    ]

    for model in models:
        print(f"Downloading {model}...")
        snapshot_download(model)
        print(f"  Done!")

    print("\nAll MLX models downloaded successfully!")
except Exception as e:
    print(f"Error downloading models: {e}")
    print("Models will be downloaded on first use.")
    sys.exit(1)
PYEOF

    local download_status=$?
    conda deactivate 2>/dev/null || true

    if [[ $download_status -eq 0 ]]; then
        success "MLX models downloaded!"
    else
        warn "Model download failed. They will be downloaded on first use."
    fi
}

download_torch_models() {
    info "Downloading PyTorch models ($SELECTED_SIZE)..."

    # Activate the right environment
    if [[ "$PLATFORM" == "linux" ]] && [[ "$LINUX_USE_CONDA" == false ]]; then
        # shellcheck source=/dev/null
        source "$USER_FILES_DIR/.venv/bin/activate"
    else
        conda activate "$TORCH_ENV_NAME"
    fi

    python3 << PYEOF
import sys
try:
    from huggingface_hub import snapshot_download

    size = "$SELECTED_SIZE"

    models = [
        f"Qwen/Qwen3-TTS-12Hz-{size}-Base",
        f"Qwen/Qwen3-TTS-12Hz-{size}-VoiceDesign",
        f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
    ]

    for model in models:
        print(f"Downloading {model}...")
        snapshot_download(model)
        print(f"  Done!")

    print("\nAll PyTorch models downloaded successfully!")
except Exception as e:
    print(f"Error downloading models: {e}")
    print("Models will be downloaded on first use.")
    sys.exit(1)
PYEOF

    local download_status=$?

    if [[ "$PLATFORM" == "linux" ]] && [[ "$LINUX_USE_CONDA" == false ]]; then
        deactivate 2>/dev/null || true
    else
        conda deactivate 2>/dev/null || true
    fi

    if [[ $download_status -eq 0 ]]; then
        success "PyTorch models downloaded!"
    else
        warn "Model download failed. They will be downloaded on first use."
    fi
}

# =============================================================================
# Section 18: Print Summary
# =============================================================================

print_summary() {
    echo ""
    echo "============================================================================="
    echo -e "${GREEN}Installation Complete!${NC}"
    echo "============================================================================="
    echo ""
    echo -e "${BOLD}Platform:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  macOS $(uname -m)"
    elif [[ "$PLATFORM" == "linux" ]]; then
        echo "  Linux ($DISTRO_ID${DISTRO_VERSION:+ $DISTRO_VERSION}, $ARCH)"
        if [[ "$HAS_NVIDIA" == true ]]; then
            echo "  GPU: $NVIDIA_GPU_NAME"
            [[ -n "$CUDA_VERSION" ]] && echo "  CUDA: $CUDA_VERSION"
        fi
    fi
    echo ""
    echo -e "${BOLD}Configuration:${NC}"
    echo "  Backend: $SELECTED_BACKEND"
    echo "  Model size: $SELECTED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo "  Quantization: $SELECTED_QUANT"
    fi
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"

    # Show activation command
    if [[ "$PLATFORM" == "linux" ]] && [[ "$LINUX_USE_CONDA" == false ]]; then
        echo "  0. Activate your environment:"
        echo -e "     ${CYAN}source $USER_FILES_DIR/.venv/bin/activate${NC}"
        echo ""
    fi

    echo "  1. Start the TTS server (loads models):"
    echo -e "     ${CYAN}tts server start${NC}"
    echo ""
    echo "  2. Generate speech:"
    echo -e "     ${CYAN}tts \"Hello, world!\" -o hello${NC}"
    echo ""
    echo "  3. Launch web interface:"
    echo -e "     ${CYAN}tts ui${NC}"
    echo ""
    echo "  4. Stop the server when done:"
    echo -e "     ${CYAN}tts server stop${NC}"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo -e "  ${CYAN}tts${NC}                - Generate speech from text"
    echo -e "  ${CYAN}tts server start${NC}   - Start the persistent TTS server"
    echo -e "  ${CYAN}tts server stop${NC}    - Stop the TTS server"
    echo -e "  ${CYAN}tts voice create${NC}   - Create a voice clone from audio"
    echo -e "  ${CYAN}tts ui${NC}             - Launch web interface"
    echo -e "  ${CYAN}tts config${NC}         - Reconfigure settings (model size, quantization)"
    echo ""
    echo -e "${BOLD}Documentation:${NC}"
    echo -e "  ${CYAN}$USER_FILES_DIR/README.md${NC}"
    echo ""
    echo -e "${BOLD}Configuration file:${NC}"
    echo -e "  ${CYAN}$USER_FILES_DIR/config.json${NC}"
    echo ""

    if [[ "$PLATFORM" == "macos" ]] && [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo -e "${YELLOW}NOTE: Open a new terminal or run 'source ~/.zshrc' to use commands.${NC}"
        echo ""
    fi
}

# =============================================================================
# Section 19: Reconfigure Only Mode
# =============================================================================

reconfigure_only() {
    echo ""
    echo "============================================================================="
    echo "Qwen3-TTS Configuration Wizard"
    echo "============================================================================="
    echo ""

    RECONFIGURE_ONLY=true

    detect_platform
    detect_optimal_settings
    run_config_wizard
    create_config

    echo ""
    success "Configuration updated!"
    echo ""
    echo "Restart the TTS server to apply changes:"
    echo -e "  ${CYAN}tts server stop && tts server start${NC}"
    echo ""
}

# =============================================================================
# Section 20: Show Current Config
# =============================================================================

show_config() {
    echo ""
    echo "============================================================================="
    echo "Qwen3-TTS Configuration"
    echo "============================================================================="
    echo ""

    detect_platform
    detect_optimal_settings

    echo -e "${BOLD}Hardware:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  Platform: macOS"
        echo "  Chip: $(uname -m)"
    elif [[ "$PLATFORM" == "linux" ]]; then
        echo "  Platform: Linux ($DISTRO_ID${DISTRO_VERSION:+ $DISTRO_VERSION})"
        echo "  Arch: $ARCH"
        if [[ "$HAS_NVIDIA" == true ]]; then
            echo "  GPU: $NVIDIA_GPU_NAME"
            [[ -n "$CUDA_VERSION" ]] && echo "  CUDA: $CUDA_VERSION"
            [[ "$VRAM_GB" -gt 0 ]] && echo "  VRAM: ${VRAM_GB}GB"
        fi
    fi
    echo "  RAM:  ${RAM_GB}GB"
    echo ""

    echo -e "${BOLD}Recommended:${NC}"
    echo "  Backend: $RECOMMENDED_BACKEND"
    echo "  Model size: $RECOMMENDED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo "  Quantization: $RECOMMENDED_QUANT"
    fi
    echo ""

    CONFIG_FILE="$USER_FILES_DIR/config.json"
    if [[ -f "$CONFIG_FILE" ]]; then
        echo -e "${BOLD}Current config.json:${NC}"
        CURRENT_BACKEND=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('advanced',{}).get('backend','torch'))" 2>/dev/null || echo "unknown")
        CURRENT_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('advanced',{}).get('model_size','1.7B'))" 2>/dev/null || echo "unknown")
        CURRENT_QUANT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('advanced',{}).get('mlx_quantization','8bit'))" 2>/dev/null || echo "unknown")
        echo "  Backend: $CURRENT_BACKEND"
        echo "  Model size: $CURRENT_SIZE"
        echo "  Quantization: $CURRENT_QUANT"
    else
        echo -e "${YELLOW}No config.json found. Run install.sh to create one.${NC}"
    fi
    echo ""
}

# =============================================================================
# Section 21: Dry Run Mode
# =============================================================================

dry_run() {
    detect_platform
    detect_optimal_settings

    echo ""
    echo "============================================================================="
    echo "DRY RUN — Preview of installation steps"
    echo "============================================================================="
    echo ""
    echo -e "${BOLD}Platform:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  macOS $(uname -m), ${RAM_GB}GB RAM"
    elif [[ "$PLATFORM" == "linux" ]]; then
        echo "  Linux ($DISTRO_ID${DISTRO_VERSION:+ $DISTRO_VERSION}, $ARCH), ${RAM_GB}GB RAM"
        if [[ "$HAS_NVIDIA" == true ]]; then
            echo "  GPU: $NVIDIA_GPU_NAME"
            [[ -n "$CUDA_VERSION" ]] && echo "  CUDA: $CUDA_VERSION"
            [[ "$VRAM_GB" -gt 0 ]] && echo "  VRAM: ${VRAM_GB}GB"
        else
            echo "  GPU: None (CPU only)"
        fi
    fi
    echo ""
    echo -e "${BOLD}Recommended settings:${NC}"
    echo "  Backend: $RECOMMENDED_BACKEND"
    echo "  Model size: $RECOMMENDED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo "  Quantization: $RECOMMENDED_QUANT"
    fi
    if [[ "$PLATFORM" == "linux" ]]; then
        echo "  dtype: $(get_linux_dtype)"
        local tq
        tq=$(get_linux_torch_quant)
        [[ "$tq" != "none" ]] && echo "  Torch quantization: $tq"
    fi
    echo ""
    echo -e "${BOLD}Installation steps:${NC}"

    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  1. Check prerequisites (conda installed, disk space)"
        echo "  2. Run configuration wizard"
        if [[ "$IS_INTEL" == true ]]; then
            echo "  3. Create PyTorch environment '$TORCH_ENV_NAME' (required for Intel)"
        else
            echo "  3. Create MLX environment '$MLX_ENV_NAME' (Apple Silicon default)"
            echo "     Optional: Create PyTorch fallback environment"
        fi
        echo "  4. Create directories ($USER_FILES_DIR, voice_prompts/)"
        echo "  5. Create config.json with selected settings"
        if [[ "$IS_INTEL" == true ]]; then
            echo "  6. CLI installed via pip entry point (use: conda activate $TORCH_ENV_NAME && tts)"
        else
            echo "  6. CLI installed via pip entry point (use: conda activate $MLX_ENV_NAME && tts)"
        fi
        echo "  7. Optional: Pre-download models"

    elif [[ "$PLATFORM" == "linux" ]]; then
        echo "  1. Check prerequisites (system deps, Python/conda, disk space)"
        echo "  2. Install system dependencies (ffmpeg, libsndfile, rubberband)"
        echo "  3. Run configuration wizard"
        if command -v conda &>/dev/null; then
            echo "  4. Create conda environment '$TORCH_ENV_NAME'"
        else
            echo "  4. Create Python venv at $USER_FILES_DIR/.venv"
        fi
        local idx_url
        idx_url=$(get_torch_index_url)
        echo "  5. Install PyTorch (index: $idx_url)"
        echo "  6. Create directories ($USER_FILES_DIR, voice_prompts/)"
        echo "  7. Create config.json with platform-aware defaults"
        echo "  8. Optional: Pre-download models"
    fi

    echo ""
    echo "Run without --dry-run to perform installation."
    echo ""
}

# =============================================================================
# Section 22: Main
# =============================================================================

main() {
    echo ""
    echo "============================================================================="
    echo "Qwen3-TTS Installation Script"
    echo "============================================================================="
    echo ""

    # Parse arguments
    case "$1" in
        --dry-run)
            dry_run
            exit 0
            ;;
        --reconfigure)
            reconfigure_only
            exit 0
            ;;
        --show)
            show_config
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Preview installation steps without executing"
            echo "  --reconfigure  Update config.json with new settings (wizard)"
            echo "  --show         Show current vs recommended settings"
            echo "  --help         Show this help message"
            echo ""
            echo "Supported platforms: macOS (Apple Silicon, Intel), Linux (NVIDIA GPU, CPU)"
            echo ""
            exit 0
            ;;
    esac

    detect_platform
    check_prerequisites
    detect_optimal_settings
    run_config_wizard
    create_directories

    # Create environments based on platform and selection
    if [[ "$PLATFORM" == "macos" ]]; then
        if [[ "$IS_INTEL" == true ]]; then
            # Intel: torch only
            create_torch_env
        else
            # Apple Silicon: MLX first, then optionally torch
            if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
                create_mlx_env
                create_torch_env  # Optional fallback
            else
                create_torch_env  # User selected torch
            fi
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        # Linux: always torch (handles conda vs venv internally)
        create_torch_env
    fi

    create_config
    update_path
    download_models
    print_summary
}

main "$@"
