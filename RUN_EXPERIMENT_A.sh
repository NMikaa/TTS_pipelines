#!/bin/bash
# Experiment A: Ready-to-run script with W&B pre-configured
# This script handles everything: W&B setup, codec extraction, and training

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}EXPERIMENT A: GEORGIAN TTS FINE-TUNING${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo

# ============================================================================
# STEP 1: Verify Environment
# ============================================================================

echo -e "${BLUE}STEP 1: Verifying environment...${NC}"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${python_version}${NC}"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch not installed${NC}"
    echo "Install: pip install torch torchaudio"
    exit 1
fi
torch_version=$(python3 -c "import torch; print(torch.__version__)")
echo -e "${GREEN}✓ PyTorch ${torch_version}${NC}"

# Check NeMo
if ! python3 -c "import nemo" 2>/dev/null; then
    echo -e "${RED}❌ NeMo not installed${NC}"
    echo "Install: pip install nemo_toolkit[tts]"
    exit 1
fi
echo -e "${GREEN}✓ NeMo installed${NC}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected${NC}"
    exit 1
fi
gpu_count=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}✓ ${gpu_count} GPU(s) available${NC}"

# Check CUDA
cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
echo -e "${GREEN}✓ CUDA ${cuda_version}${NC}"

echo

# ============================================================================
# STEP 2: Configure W&B
# ============================================================================

echo -e "${BLUE}STEP 2: Configuring W&B...${NC}"
echo

# Set W&B API key (pre-configured)
export WANDB_API_KEY="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"

# Test W&B login
if python3 -c "import wandb; wandb.login()" 2>/dev/null; then
    echo -e "${GREEN}✓ W&B authenticated${NC}"
    echo "  Project: georgian-tts"
    echo "  Run: magpie-experiment-a"
    echo "  Dashboard: https://wandb.ai/"
else
    echo -e "${RED}⚠️  W&B login failed (optional, training will still work)${NC}"
fi

echo

# ============================================================================
# STEP 3: Codec Extraction
# ============================================================================

echo -e "${BLUE}STEP 3: Extracting NanoCodec tokens...${NC}"
echo "Timeline: 2-3 hours (GPU-intensive)"
echo "Logs: data/saba_experiment_a/logs/codec_preparation_*.log"
echo

cd pipelines/magpie_tts

# Check if codec extraction already done
if [ -d "../../data/saba_experiment_a/codec_codes" ] && [ "$(ls -1 ../../data/saba_experiment_a/codec_codes/ 2>/dev/null | wc -l)" -gt 0 ]; then
    codec_count=$(ls -1 ../../data/saba_experiment_a/codec_codes/ | wc -l)
    echo -e "${GREEN}✓ Codec extraction already complete (${codec_count} files)${NC}"
    echo "  Skipping codec extraction..."
    echo
else
    echo "Starting codec extraction..."
    echo
    python3 prepare_experiment_a_codec.py
    echo
fi

# ============================================================================
# STEP 4: Verify Codec Extraction
# ============================================================================

echo -e "${BLUE}STEP 4: Verifying codec extraction...${NC}"
echo

codec_count=$(ls -1 ../../data/saba_experiment_a/codec_codes/ | wc -l)
nemo_train_count=$(wc -l < ../../data/saba_experiment_a/train_manifest_nemo.json)
nemo_val_count=$(wc -l < ../../data/saba_experiment_a/val_manifest_nemo.json)

echo "Codec files: ${codec_count} (expected ~186k)"
echo "NeMo train manifest: ${nemo_train_count} (expected 182406)"
echo "NeMo val manifest: ${nemo_val_count} (expected 3770)"
echo

if [ "${codec_count}" -gt 100000 ]; then
    echo -e "${GREEN}✓ Codec extraction verified${NC}"
else
    echo -e "${RED}❌ Codec extraction may have failed${NC}"
    exit 1
fi

echo

# ============================================================================
# STEP 5: Launch Training
# ============================================================================

echo -e "${BLUE}STEP 5: Launching training...${NC}"
echo "Timeline: 2-3 weeks (2 GPU DDP)"
echo "Logs: data/saba_experiment_a/logs/training_*.log"
echo

python3 run_experiment_a.py

# ============================================================================
# STEP 6: Post-Training
# ============================================================================

echo
echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}✓ EXPERIMENT A COMPLETE${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo
echo "Training outputs:"
echo "  Checkpoints: exp/experiment_a_punctuated_joint_training/version_0/checkpoints/"
echo "  Logs: exp/experiment_a_punctuated_joint_training/version_0/"
echo
echo "Next steps:"
echo "  1. Evaluate on FLEURS test set (CER < 3% target)"
echo "  2. Test zero-shot voice cloning (CER < 6% target)"
echo "  3. Check W&B dashboard for training curves"
echo
echo "References:"
echo "  - Quick start: EXPERIMENT_A_QUICK_START.md"
echo "  - Monitoring: EXPERIMENT_A_LOGGING.md"
echo "  - Troubleshooting: EXPERIMENT_A_SETUP.md"
echo
