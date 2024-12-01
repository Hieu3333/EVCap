# Use an official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies required for the Google Cloud SDK and system libraries for pycocotools
RUN apt-get update && \
    apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-dev \
    build-essential \
    libjpeg-dev \
    libpng-dev && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk

# Install Conda
RUN curl -sS https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

# Set PATH to include conda
ENV PATH="/opt/conda/bin:$PATH"

# Install pycocotools before creating the Conda environment
RUN pip install pycocotools

# Copy the environment.yaml into the container
COPY environment.yaml .

# Create the Conda environment using the environment.yaml
RUN conda env create -f environment.yaml

# Activate the environment (for subsequent commands)
SHELL ["conda", "run", "-n", "env_evcap", "/bin/bash", "-c"]

# Install Google Cloud Storage Python client
RUN pip install google-cloud-storage

# Copy your project files into the container
COPY . /app

# Set the default environment to use
ENV PATH /opt/conda/envs/env_evcap/bin:$PATH

# Run your main script by default
CMD ["python", "generate.py"]
