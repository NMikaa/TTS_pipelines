#!/bin/bash
# WandB Setup Script for Experiment A

echo "========================================================================"
echo "W&B SETUP FOR EXPERIMENT A"
echo "========================================================================"
echo

# Check if W&B CLI is installed
if ! command -v wandb &> /dev/null; then
    echo "❌ wandb CLI not found. Installing..."
    pip install wandb
fi

# Prompt user for API key
echo "To log into W&B, you need your API key."
echo "Get it from: https://wandb.ai/authorize"
echo
read -p "Enter your W&B API key (or press Enter to skip): " wandb_key

if [ -z "$wandb_key" ]; then
    echo "⚠️  Skipping W&B login. You can login later with: wandb login"
    exit 0
fi

# Login to W&B
wandb login "$wandb_key"

echo
echo "✓ W&B configured successfully!"
echo
echo "Project: georgian-tts"
echo "Run name: magpie-experiment-a"
echo
echo "Monitor at: https://wandb.ai/"
