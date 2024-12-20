#!/bin/bash

# Exit the script on error
set -e
cd /workspace
# Step 1: Install Miniconda
echo "Installing Miniconda..."
# Download Miniconda (Linux version)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# Install Miniconda silently
bash miniconda.sh -b -p $HOME/miniconda
# Initialize conda (adds it to the PATH)
$HOME/miniconda/bin/conda init
# Reload shell
source ~/.bashrc

# Step 2: Clone the remote repository to /workspace
echo "Cloning the repository..."
# Change to the workspace directory (ensure you have access to it)
git clone https://github.com/Hieu3333/EVCap.git
cd EVCap

# Step 3: Create a conda environment from environment.yaml inside /workspace
echo "Creating conda environment from environment.yaml..."
# Create the environment inside /workspace
conda env create -f environment.yaml -p /workspace/env_evcap
# Activate the environment
source activate /workspace/env_evcap

echo "Environment setup complete."
export HF_HOME=/workspace/huggingface_cache


# echo "Installing Google Cloud SDK..."
# sudo apt-get update -y
# # Install Google Cloud SDK
# sudo apt-get install google-cloud-sdk -y
# Install Google Cloud Storage library via conda
conda install -c conda-forge google-cloud-storage

echo "Downloading Coco dataset"
wget http://images.cocodataset.org/zips/train2014.zip
unzip -q train2014.zip -d data/coco/coco2014

