#!/bin/bash
# Qwen3-TTS Installation Script
# MLX-First architecture with smart hardware detection

set -e

# =============================================================================
# Configuration
# =============================================================================

MLX_ENV_NAME="qwen3-tts-mlx"
TORCH_ENV_NAME="qwen3-tts"
PYTHON_VERSION="3.11"
USER_FILES_DIR="$HOME/Qwen3-TTS_UserFiles"
BIN_DIR="$HOME/bin"
VOICE_PROMPTS_DIR="$USER_FILES_DIR/voice_prompts"
MIN_DISK_SPACE_GB=15

# Detected optimal settings (set by detect_optimal_settings)
RECOMMENDED_BACKEND=""
RECOMMENDED_SIZE=""
RECOMMENDED_QUANT=""
IS_INTEL=false
RAM_GB=0

# User-selected settings
SELECTED_BACKEND=""
SELECTED_SIZE=""
SELECTED_QUANT=""

# =============================================================================
# Color Output Functions
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
# Hardware Detection
# =============================================================================

detect_optimal_settings() {
    step "Detecting hardware..."

    ARCH=$(uname -m)

    # Check for Apple Silicon vs Intel
    if [[ "$ARCH" == "arm64" ]]; then
        RECOMMENDED_BACKEND="mlx"
        IS_INTEL=false
        success "Apple Silicon detected (arm64) - MLX backend available"
    else
        RECOMMENDED_BACKEND="torch"
        IS_INTEL=true
        warn "Intel Mac detected ($ARCH) - MLX not available, using PyTorch backend"
    fi

    # Check RAM (macOS)
    RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
    if [[ -z "$RAM_GB" ]] || [[ "$RAM_GB" -eq 0 ]]; then
        RAM_GB=16  # Default assumption
    fi

    # Recommend settings based on RAM
    if [[ "$RAM_GB" -lt 16 ]]; then
        RECOMMENDED_SIZE="0.6B"
        RECOMMENDED_QUANT="4bit"
        info "RAM: ${RAM_GB}GB - Recommending smaller models (0.6B, 4bit)"
    else
        RECOMMENDED_SIZE="1.7B"
        RECOMMENDED_QUANT="8bit"
        info "RAM: ${RAM_GB}GB - Recommending standard models (1.7B, 8bit)"
    fi

    # Set defaults
    SELECTED_BACKEND="$RECOMMENDED_BACKEND"
    SELECTED_SIZE="$RECOMMENDED_SIZE"
    SELECTED_QUANT="$RECOMMENDED_QUANT"
}

# =============================================================================
# Interactive Configuration Wizard
# =============================================================================

run_config_wizard() {
    step "Configuration Wizard"

    echo ""
    echo -e "${BOLD}Hardware Detection${NC}"
    echo "=================="
    echo "  Chip: $(uname -m)"
    echo "  RAM:  ${RAM_GB}GB"
    echo ""

    echo -e "${BOLD}Recommended Configuration${NC}"
    echo "========================="
    if [[ "$IS_INTEL" == true ]]; then
        echo "  Backend: torch (Intel Mac - MLX not available)"
    else
        echo "  Backend: $RECOMMENDED_BACKEND (native Apple Silicon)"
    fi
    echo "  Model size: $RECOMMENDED_SIZE"
    if [[ "$IS_INTEL" == false ]]; then
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

    # Backend selection (only for Apple Silicon)
    if [[ "$IS_INTEL" == false ]]; then
        echo -e "${BOLD}Backend${NC}"
        echo "  1. mlx - Apple Silicon native, lower thermals (recommended)"
        echo "  2. torch - PyTorch/MPS, if you prefer or need torch compatibility"
        read -p "Select [1]: " BACKEND_CHOICE
        BACKEND_CHOICE=${BACKEND_CHOICE:-1}

        if [[ "$BACKEND_CHOICE" == "2" ]]; then
            SELECTED_BACKEND="torch"
        else
            SELECTED_BACKEND="mlx"
        fi
    else
        echo -e "${YELLOW}Backend: torch (Intel Mac - no other options)${NC}"
        SELECTED_BACKEND="torch"
    fi

    echo ""

    # Model size selection
    echo -e "${BOLD}Model Size${NC}"
    echo "  1. 1.7B - Higher quality, ~3.5GB per model"
    echo "  2. 0.6B - Faster (~40%), lower memory, good for <16GB RAM"
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
        echo "  1. 8bit - Good balance of quality and size (recommended)"
        echo "  2. 4bit - Smallest, fastest, slightly lower quality"
        echo "  3. bf16 - Full precision, highest quality, largest size"
        read -p "Select [1]: " QUANT_CHOICE
        QUANT_CHOICE=${QUANT_CHOICE:-1}

        case "$QUANT_CHOICE" in
            2) SELECTED_QUANT="4bit" ;;
            3) SELECTED_QUANT="bf16" ;;
            *) SELECTED_QUANT="8bit" ;;
        esac
    fi
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    step "Checking prerequisites..."

    # Check macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        error "This script is designed for macOS only."
        exit 1
    fi
    info "macOS detected"

    # Note: We now support both Apple Silicon AND Intel
    # Intel gets torch-only, Apple Silicon gets MLX as default

    # Check for conda
    if ! command -v conda &> /dev/null; then
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

    # Install rubberband for high-quality audio speed/pitch processing
    if command -v brew &> /dev/null; then
        if ! brew list rubberband &> /dev/null 2>&1; then
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

    # Check disk space
    AVAILABLE_GB=$(df -g "$HOME" | awk 'NR==2 {print $4}')
    if [[ "$AVAILABLE_GB" -lt "$MIN_DISK_SPACE_GB" ]]; then
        warn "Low disk space: ${AVAILABLE_GB}GB available (recommended: ${MIN_DISK_SPACE_GB}GB+)"
        warn "Models require ~10GB of disk space."
        read -p "Continue anyway? [y/N]: " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    else
        info "Disk space: ${AVAILABLE_GB}GB available"
    fi

    success "All prerequisites met!"
}

# =============================================================================
# Source Conda
# =============================================================================

source_conda() {
    if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        CONDA_SH=$(find "$HOME" -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null | head -1)
        if [[ -n "$CONDA_SH" ]]; then
            source "$CONDA_SH"
        else
            error "Could not find conda.sh. Please ensure conda is properly installed."
            exit 1
        fi
    fi
}

# =============================================================================
# Create MLX Conda Environment (Primary for Apple Silicon)
# =============================================================================

create_mlx_env() {
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
            pip install --upgrade -r "$USER_FILES_DIR/requirements-mlx.txt"
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
    pip install -r "$USER_FILES_DIR/requirements-mlx.txt"

    conda deactivate 2>/dev/null || true

    success "MLX environment created and packages installed!"
}

# =============================================================================
# Create Torch Conda Environment (Fallback / Intel Primary)
# =============================================================================

create_torch_env() {
    # For Intel: always required
    # For Apple Silicon: optional (if user wants torch fallback or selected torch)

    if [[ "$IS_INTEL" == true ]]; then
        step "Setting up PyTorch environment: $TORCH_ENV_NAME (required for Intel Mac)..."
    elif [[ "$SELECTED_BACKEND" == "torch" ]]; then
        step "Setting up PyTorch environment: $TORCH_ENV_NAME..."
    else
        # Apple Silicon with MLX - offer torch as optional fallback
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
            pip install --upgrade qwen-tts flask librosa soundfile numpy pydub requests gradio watchdog
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
    pip install qwen-tts flask librosa soundfile numpy pydub requests gradio watchdog

    conda deactivate 2>/dev/null || true

    success "PyTorch environment created and packages installed!"
}

# =============================================================================
# Create Directories
# =============================================================================

create_directories() {
    step "Creating directories..."

    mkdir -p "$USER_FILES_DIR"
    mkdir -p "$VOICE_PROMPTS_DIR"
    mkdir -p "$BIN_DIR"

    success "Directories created:"
    info "  $USER_FILES_DIR"
    info "  $VOICE_PROMPTS_DIR"
    info "  $BIN_DIR"
}

# =============================================================================
# Create Config File
# =============================================================================

create_config() {
    step "Creating configuration file..."

    CONFIG_FILE="$USER_FILES_DIR/config.json"

    if [[ -f "$CONFIG_FILE" ]] && [[ "$RECONFIGURE_ONLY" != true ]]; then
        warn "config.json already exists."
        read -p "Overwrite with new settings? [y/N]: " OVERWRITE
        if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
            info "Keeping existing config.json"
            return
        fi
    fi

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
    "dtype": "bfloat16",
    "backend": "$SELECTED_BACKEND",
    "mlx_quantization": "$SELECTED_QUANT",
    "model_size": "$SELECTED_SIZE"
  },
  "generation": {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "seed": null,
    "max_chunk_chars": 500
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
}

# =============================================================================
# Create Wrapper Scripts
# =============================================================================

create_wrapper_scripts() {
    step "Installing wrapper scripts to ~/bin/..."

    SCRIPT_SRC="$USER_FILES_DIR/bin"

    if [[ ! -d "$SCRIPT_SRC" ]]; then
        error "bin/ directory not found in $USER_FILES_DIR"
        error "Please ensure you cloned the complete repository."
        exit 1
    fi

    for script in changeVoice startTTSServer stopTTSServer createVoice ttsUI configureTTS; do
        if [[ -f "$SCRIPT_SRC/$script" ]]; then
            cp "$SCRIPT_SRC/$script" "$BIN_DIR/$script"
            info "Installed $script"
        else
            if [[ "$script" != "configureTTS" ]]; then
                warn "Script $script not found in $SCRIPT_SRC, skipping."
            fi
        fi
    done

    success "Wrapper scripts installed"
}

# =============================================================================
# Set Permissions
# =============================================================================

set_permissions() {
    step "Setting executable permissions..."

    for script in changeVoice startTTSServer stopTTSServer createVoice ttsUI configureTTS; do
        if [[ -f "$BIN_DIR/$script" ]]; then
            chmod 755 "$BIN_DIR/$script"
        fi
    done

    success "Permissions set"
}

# =============================================================================
# Update PATH
# =============================================================================

update_path() {
    step "Checking PATH configuration..."

    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        warn "~/bin is not in your PATH."

        if [[ "$SHELL" == *"zsh"* ]]; then
            SHELL_RC="$HOME/.zshrc"
        else
            SHELL_RC="$HOME/.bashrc"
        fi

        echo ""
        read -p "Add ~/bin to PATH in $SHELL_RC? [Y/n]: " ADD_PATH
        ADD_PATH=${ADD_PATH:-Y}

        if [[ "$ADD_PATH" =~ ^[Yy]$ ]]; then
            echo '' >> "$SHELL_RC"
            echo '# Added by Qwen3-TTS installer' >> "$SHELL_RC"
            echo 'export PATH="$HOME/bin:$PATH"' >> "$SHELL_RC"
            success "Added ~/bin to PATH in $SHELL_RC"
            warn "Run 'source $SHELL_RC' or open a new terminal to use the commands."
        else
            warn "Please add ~/bin to your PATH manually:"
            echo '  export PATH="$HOME/bin:$PATH"'
        fi
    else
        success "~/bin is already in PATH"
    fi
}

# =============================================================================
# Download Models
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

    source_conda

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

    conda deactivate 2>/dev/null || true

    if [[ $? -eq 0 ]]; then
        success "MLX models downloaded!"
    else
        warn "Model download failed. They will be downloaded on first use."
    fi
}

download_torch_models() {
    info "Downloading PyTorch models ($SELECTED_SIZE)..."
    conda activate "$TORCH_ENV_NAME"

    python << PYEOF
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

    conda deactivate 2>/dev/null || true

    if [[ $? -eq 0 ]]; then
        success "PyTorch models downloaded!"
    else
        warn "Model download failed. They will be downloaded on first use."
    fi
}

# =============================================================================
# Print Summary
# =============================================================================

print_summary() {
    echo ""
    echo "============================================================================="
    echo -e "${GREEN}Installation Complete!${NC}"
    echo "============================================================================="
    echo ""
    echo -e "${BOLD}Configuration:${NC}"
    echo "  Backend: $SELECTED_BACKEND"
    echo "  Model size: $SELECTED_SIZE"
    if [[ "$SELECTED_BACKEND" == "mlx" ]]; then
        echo "  Quantization: $SELECTED_QUANT"
    fi
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo "  1. Start the TTS server (loads models):"
    echo "     ${CYAN}startTTSServer${NC}"
    echo ""
    echo "  2. Generate speech:"
    echo "     ${CYAN}changeVoice \"Hello, world!\" -o hello${NC}"
    echo ""
    echo "  3. Launch web interface:"
    echo "     ${CYAN}ttsUI${NC}"
    echo ""
    echo "  4. Stop the server when done:"
    echo "     ${CYAN}stopTTSServer${NC}"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo "  ${CYAN}changeVoice${NC}     - Generate speech from text"
    echo "  ${CYAN}startTTSServer${NC}  - Start the persistent TTS server"
    echo "  ${CYAN}stopTTSServer${NC}   - Stop the TTS server"
    echo "  ${CYAN}createVoice${NC}     - Create a voice clone from audio"
    echo "  ${CYAN}ttsUI${NC}           - Launch web interface"
    echo "  ${CYAN}configureTTS${NC}    - Reconfigure settings (model size, quantization)"
    echo ""
    echo -e "${BOLD}Documentation:${NC}"
    echo "  ${CYAN}$USER_FILES_DIR/README.md${NC}"
    echo ""
    echo -e "${BOLD}Configuration file:${NC}"
    echo "  ${CYAN}$USER_FILES_DIR/config.json${NC}"
    echo ""

    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo -e "${YELLOW}NOTE: Open a new terminal or run 'source ~/.zshrc' to use commands.${NC}"
        echo ""
    fi
}

# =============================================================================
# Reconfigure Only Mode
# =============================================================================

reconfigure_only() {
    echo ""
    echo "============================================================================="
    echo "Qwen3-TTS Configuration Wizard"
    echo "============================================================================="
    echo ""

    RECONFIGURE_ONLY=true

    detect_optimal_settings
    run_config_wizard
    create_config

    echo ""
    success "Configuration updated!"
    echo ""
    echo "Restart the TTS server to apply changes:"
    echo "  ${CYAN}stopTTSServer && startTTSServer${NC}"
    echo ""
}

# =============================================================================
# Show Current Config
# =============================================================================

show_config() {
    echo ""
    echo "============================================================================="
    echo "Qwen3-TTS Configuration"
    echo "============================================================================="
    echo ""

    detect_optimal_settings

    echo -e "${BOLD}Hardware:${NC}"
    echo "  Chip: $(uname -m)"
    echo "  RAM:  ${RAM_GB}GB"
    echo ""

    echo -e "${BOLD}Recommended:${NC}"
    echo "  Backend: $RECOMMENDED_BACKEND"
    echo "  Model size: $RECOMMENDED_SIZE"
    echo "  Quantization: $RECOMMENDED_QUANT"
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
# Dry Run Mode
# =============================================================================

dry_run() {
    detect_optimal_settings

    echo ""
    echo "============================================================================="
    echo "DRY RUN - Preview of installation steps"
    echo "============================================================================="
    echo ""
    echo "Hardware detected: $(uname -m), ${RAM_GB}GB RAM"
    echo ""
    echo "Recommended settings:"
    echo "  Backend: $RECOMMENDED_BACKEND"
    echo "  Model size: $RECOMMENDED_SIZE"
    echo "  Quantization: $RECOMMENDED_QUANT"
    echo ""
    echo "Installation steps:"
    echo "  1. Check prerequisites (conda installed, disk space)"
    echo "  2. Run configuration wizard"
    if [[ "$IS_INTEL" == true ]]; then
        echo "  3. Create PyTorch environment '$TORCH_ENV_NAME' (required for Intel)"
    else
        echo "  3. Create MLX environment '$MLX_ENV_NAME' (Apple Silicon default)"
        echo "     Optional: Create PyTorch fallback environment"
    fi
    echo "  4. Create directories ($USER_FILES_DIR, $BIN_DIR)"
    echo "  5. Create config.json with selected settings"
    echo "  6. Install wrapper scripts to ~/bin/"
    echo "  7. Optional: Pre-download models"
    echo ""
    echo "Run without --dry-run to perform installation."
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo "============================================================================="
    echo "Qwen3-TTS Installation Script (MLX-First)"
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
            exit 0
            ;;
    esac

    check_prerequisites
    detect_optimal_settings
    run_config_wizard
    create_directories

    # Create environments based on selection
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

    create_config
    create_wrapper_scripts
    set_permissions
    update_path
    download_models
    print_summary
}

main "$@"
