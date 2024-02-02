#!/bin/bash

# Clone the repository
git clone https://github.com/haofanwang/inswapper.git
cd inswapper

# Install Git LFS
git lfs install

# Clone the Hugging Face model
git clone https://huggingface.co/spaces/sczhou/CodeFormer

# Move back to the parent directory
cd ..

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements_versions.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy directories
cp -r inswapper/CodeFormer/CodeFormer/basicsr venv/lib/python*/site-packages/
cp -r inswapper/CodeFormer/CodeFormer/facelib venv/lib/python*/site-packages/

# Create a directory for checkpoints
mkdir -p inswapper/checkpoints

# Download the ONNX model
wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx -O inswapper/checkpoints/inswapper_128.onnx
