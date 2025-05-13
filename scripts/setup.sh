#!/bin/bash


set -e
cd /workspace
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init
source ~/.bashrc


echo "Cloning the repository..."
git clone https://github.com/NHH-123/EVCap.git
cd EVCap

echo "Creating conda environment from environment.yaml..."
conda env create -f environment.yaml -p /workspace/env_evcap

source activate /workspace/env_evcap

echo "Environment setup complete."
export HF_HOME=/workspace/huggingface_cache
python -m spacy download en_core_web_sm
apt update
apt install openjdk-11-jre


# echo "Installing Google Cloud SDK..."
# sudo apt-get update -y
# # Install Google Cloud SDK
# sudo apt-get install google-cloud-sdk -y
# Install Google Cloud Storage library via conda
conda install -c conda-forge google-cloud-storage

echo "Downloading Coco dataset"
wget http://images.cocodataset.org/zips/train2014.zip
unzip -q train2014.zip -d data/coco/coco2014



