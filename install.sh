#!/bin/bash
# Qwen3-TTS Installation Script
# Automated setup for macOS with Apple Silicon

set -e

# =============================================================================
# Configuration
# =============================================================================

CONDA_ENV_NAME="qwen3-tts"
PYTHON_VERSION="3.11"
USER_FILES_DIR="$HOME/Qwen3-TTS_UserFiles"
BIN_DIR="$HOME/bin"
VOICE_PROMPTS_DIR="$USER_FILES_DIR/voice_prompts"
MIN_DISK_SPACE_GB=15

# =============================================================================
# Color Output Functions
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
step() { echo -e "\n${CYAN}==> $1${NC}"; }

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

    # Check Apple Silicon (arm64)
    if [[ "$(uname -m)" != "arm64" ]]; then
        error "This script requires Apple Silicon (M1/M2/M3/M4)."
        error "Intel Macs are not supported due to MPS requirements."
        exit 1
    fi
    success "Apple Silicon detected"

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
# Create Conda Environment
# =============================================================================

create_conda_env() {
    step "Setting up conda environment: $CONDA_ENV_NAME..."

    # Source conda
    if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        # Try to find conda.sh
        CONDA_SH=$(find "$HOME" -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null | head -1)
        if [[ -n "$CONDA_SH" ]]; then
            source "$CONDA_SH"
        else
            error "Could not find conda.sh. Please ensure conda is properly installed."
            exit 1
        fi
    fi

    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        warn "Environment '$CONDA_ENV_NAME' already exists."
        read -p "Recreate environment? This will delete the existing one. [y/N]: " RECREATE
        if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
            info "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            info "Keeping existing environment. Updating packages..."
            conda activate "$CONDA_ENV_NAME"
            pip install --upgrade qwen-tts flask librosa soundfile numpy pydub requests gradio watchdog
            success "Packages updated!"
            return
        fi
    fi

    info "Creating new conda environment with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y

    info "Activating environment..."
    conda activate "$CONDA_ENV_NAME"

    info "Installing PyTorch with MPS support..."
    pip install torch torchvision torchaudio

    info "Installing Qwen3-TTS and dependencies..."
    pip install qwen-tts flask librosa soundfile numpy pydub requests gradio watchdog

    success "Conda environment created and packages installed!"
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

    if [[ -f "$CONFIG_FILE" ]]; then
        warn "config.json already exists. Skipping."
        return
    fi

    cat > "$CONFIG_FILE" << 'EOF'
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
  "generation": {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "seed": null
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
  "aliases": {
    "default": {
      "prompt": "default_clone.pt",
      "preset": "consistent"
    }
  }
}
EOF

    success "Created config.json with default settings"
}

# =============================================================================
# Create Wrapper Scripts
# =============================================================================

create_wrapper_scripts() {
    step "Creating wrapper scripts in ~/bin/..."

    # Determine conda path
    if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        CONDA_PATH="$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        CONDA_PATH=$(find "$HOME" -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null | head -1)
    fi

    # changeVoice
    cat > "$BIN_DIR/changeVoice" << EOF
#!/bin/bash
# Qwen3-TTS voice generation wrapper
# Handles server detection and user prompts

source $CONDA_PATH && conda activate $CONDA_ENV_NAME

# Check if server is running
SERVER_RUNNING=false
if curl -s http://127.0.0.1:5123/health > /dev/null 2>&1; then
    SERVER_RUNNING=true
fi

# If server not running, ask user if they want to start it
if [ "\$SERVER_RUNNING" = false ]; then
    echo "TTS Server is not running."
    echo "Starting the server keeps models in memory for faster generation (~2-5 sec vs ~30-60 sec)."
    read -p "Would you like to start the TTS server? [Y/n]: " START_SERVER
    START_SERVER=\${START_SERVER:-Y}

    if [[ "\$START_SERVER" =~ ^[Yy]\$ ]]; then
        echo ""
        startTTSServer
        if [ \$? -eq 0 ]; then
            SERVER_RUNNING=true
        else
            echo "Failed to start server. Falling back to local generation."
        fi
        echo ""
    fi
fi

# Run the generation script
if [ "\$SERVER_RUNNING" = true ]; then
    python ~/Qwen3-TTS_UserFiles/tts_generate.py --_server-mode "\$@"
    RESULT=\$?
else
    python ~/Qwen3-TTS_UserFiles/tts_generate.py "\$@"
    RESULT=\$?
fi

# If server was used (exit code 2), ask if user wants to stop it
if [ \$RESULT -eq 2 ]; then
    echo ""
    read -p "Would you like to stop the TTS server? [y/N]: " STOP_SERVER
    STOP_SERVER=\${STOP_SERVER:-N}

    if [[ "\$STOP_SERVER" =~ ^[Yy]\$ ]]; then
        stopTTSServer
    else
        echo "Server still running. Use 'stopTTSServer' to stop it later."
    fi
fi
EOF

    # startTTSServer
    cat > "$BIN_DIR/startTTSServer" << EOF
#!/bin/bash
# Start the TTS server in the background

PID_FILE="\$HOME/Qwen3-TTS_UserFiles/.tts_server.pid"
LOG_FILE="\$HOME/Qwen3-TTS_UserFiles/.tts_server.log"

# Check if already running
if [ -f "\$PID_FILE" ]; then
    PID=\$(cat "\$PID_FILE")
    if kill -0 "\$PID" 2>/dev/null; then
        echo "TTS Server is already running (PID: \$PID)"
        exit 0
    else
        rm -f "\$PID_FILE"
    fi
fi

echo "Starting TTS Server..."
echo "This will load the models into memory (may take 30-60 seconds)..."

# Activate conda and start server in background
source $CONDA_PATH && conda activate $CONDA_ENV_NAME

# Start server in background, redirect output to log
nohup python ~/Qwen3-TTS_UserFiles/tts_server.py > "\$LOG_FILE" 2>&1 &

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..120}; do
    if curl -s http://127.0.0.1:5123/health > /dev/null 2>&1; then
        echo "TTS Server is ready!"
        echo "Use 'changeVoice' for fast generation."
        echo "Use 'stopTTSServer' to shut down."
        exit 0
    fi
    sleep 1
    # Show progress every 10 seconds
    if [ \$((i % 10)) -eq 0 ]; then
        echo "  Still loading models... (\$i seconds)"
    fi
done

echo "Error: Server failed to start. Check log: \$LOG_FILE"
exit 1
EOF

    # stopTTSServer
    cat > "$BIN_DIR/stopTTSServer" << 'EOF'
#!/bin/bash
# Stop the TTS server

PID_FILE="$HOME/Qwen3-TTS_UserFiles/.tts_server.pid"

# Try graceful shutdown via API first
if curl -s -X POST http://127.0.0.1:5123/shutdown > /dev/null 2>&1; then
    echo "TTS Server shutting down gracefully..."
    sleep 2
fi

# Check PID file and kill if still running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID" 2>/dev/null
        sleep 1
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null
        fi
    fi
    rm -f "$PID_FILE"
fi

echo "TTS Server stopped."
EOF

    # createVoice
    cat > "$BIN_DIR/createVoice" << EOF
#!/bin/bash
# Create a new custom voice clone prompt
# Activates conda environment and runs the voice creation script

source $CONDA_PATH && conda activate $CONDA_ENV_NAME

python ~/Qwen3-TTS_UserFiles/create_custom_voice.py "\$@"
EOF

    # ttsUI
    cat > "$BIN_DIR/ttsUI" << EOF
#!/bin/bash
# Launch the Qwen3-TTS web interface

source $CONDA_PATH && conda activate $CONDA_ENV_NAME

python ~/Qwen3-TTS_UserFiles/tts_ui.py "\$@"
EOF

    success "Created wrapper scripts"
}

# =============================================================================
# Set Permissions
# =============================================================================

set_permissions() {
    step "Setting executable permissions..."

    chmod 755 "$BIN_DIR/changeVoice"
    chmod 755 "$BIN_DIR/startTTSServer"
    chmod 755 "$BIN_DIR/stopTTSServer"
    chmod 755 "$BIN_DIR/createVoice"
    chmod 755 "$BIN_DIR/ttsUI"

    success "Permissions set"
}

# =============================================================================
# Update PATH
# =============================================================================

update_path() {
    step "Checking PATH configuration..."

    # Check if ~/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        warn "~/bin is not in your PATH."

        # Determine shell config file
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
# Pre-download Models (Optional)
# =============================================================================

download_models() {
    step "Model pre-download (optional)..."

    echo ""
    echo "The Qwen3-TTS models are ~10GB total and will be downloaded on first use."
    echo "You can pre-download them now (5-10 minutes) or skip this step."
    echo ""
    read -p "Download models now? [y/N]: " DOWNLOAD
    DOWNLOAD=${DOWNLOAD:-N}

    if [[ "$DOWNLOAD" =~ ^[Yy]$ ]]; then
        info "Downloading models... This may take 5-10 minutes."

        # Activate conda
        source "$CONDA_PATH" && conda activate "$CONDA_ENV_NAME"

        python << 'PYEOF'
import sys
try:
    from huggingface_hub import snapshot_download

    models = [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    ]

    for model in models:
        print(f"Downloading {model}...")
        snapshot_download(model)
        print(f"  Done!")

    print("\nAll models downloaded successfully!")
except Exception as e:
    print(f"Error downloading models: {e}")
    print("Models will be downloaded on first use.")
    sys.exit(1)
PYEOF

        if [[ $? -eq 0 ]]; then
            success "Models downloaded!"
        else
            warn "Model download failed. They will be downloaded on first use."
        fi
    else
        info "Skipping model download. Models will be downloaded on first use."
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
    echo "Quick Start:"
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
    echo "Commands:"
    echo "  ${CYAN}changeVoice${NC}     - Generate speech from text"
    echo "  ${CYAN}startTTSServer${NC}  - Start the persistent TTS server"
    echo "  ${CYAN}stopTTSServer${NC}   - Stop the TTS server"
    echo "  ${CYAN}createVoice${NC}     - Create a voice clone from audio"
    echo "  ${CYAN}ttsUI${NC}           - Launch web interface"
    echo ""
    echo "Documentation:"
    echo "  ${CYAN}$USER_FILES_DIR/README.md${NC}"
    echo ""
    echo "Configuration:"
    echo "  ${CYAN}$USER_FILES_DIR/config.json${NC}"
    echo ""

    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo -e "${YELLOW}NOTE: Open a new terminal or run 'source ~/.zshrc' to use commands.${NC}"
        echo ""
    fi
}

# =============================================================================
# Dry Run Mode
# =============================================================================

dry_run() {
    echo ""
    echo "============================================================================="
    echo "DRY RUN - Preview of installation steps"
    echo "============================================================================="
    echo ""
    echo "1. Check prerequisites:"
    echo "   - Verify macOS + Apple Silicon"
    echo "   - Check conda is installed"
    echo "   - Warn if disk space < ${MIN_DISK_SPACE_GB}GB"
    echo ""
    echo "2. Create conda environment '$CONDA_ENV_NAME' with:"
    echo "   - Python $PYTHON_VERSION"
    echo "   - torch, qwen-tts, flask, librosa, soundfile, numpy"
    echo "   - pydub, requests, gradio, watchdog"
    echo ""
    echo "3. Create directories:"
    echo "   - $USER_FILES_DIR"
    echo "   - $VOICE_PROMPTS_DIR"
    echo "   - $BIN_DIR"
    echo ""
    echo "4. Create config.json with sensible defaults"
    echo ""
    echo "5. Create wrapper scripts in ~/bin/:"
    echo "   - changeVoice"
    echo "   - startTTSServer"
    echo "   - stopTTSServer"
    echo "   - createVoice"
    echo "   - ttsUI"
    echo ""
    echo "6. Set executable permissions (chmod 755)"
    echo ""
    echo "7. Optional: Pre-download models (~10GB)"
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
    echo "Qwen3-TTS Installation Script"
    echo "============================================================================="
    echo ""

    # Check for dry run
    if [[ "$1" == "--dry-run" ]]; then
        dry_run
        exit 0
    fi

    check_prerequisites
    create_directories
    create_conda_env
    create_config
    create_wrapper_scripts
    set_permissions
    update_path
    download_models
    print_summary
}

main "$@"
