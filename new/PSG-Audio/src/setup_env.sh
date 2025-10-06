#!/bin/bash

# Print initial state
echo "🔍 Initial Environment Check:"
nvidia-smi

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Set specific TensorFlow configurations
export TF_CUDA_PATHS="/usr/local/cuda,/usr/local/cuda-12.9,/usr/lib/x86_64-linux-gnu"
export TF_CUDA_VERSION="12.9"
export TF_CUDNN_VERSION="8"

# Print current configuration
echo -e "\n📊 Environment Setup:"
echo "----------------"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH includes CUDA: $(echo $PATH | grep -q cuda && echo 'Yes' || echo 'No')"
echo "LD_LIBRARY_PATH includes CUDA: $(echo $LD_LIBRARY_PATH | grep -q cuda && echo 'Yes' || echo 'No')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check for cuDNN
echo -e "\n🔍 Checking cuDNN Installation:"
if [ -f "/usr/local/cuda/include/cudnn.h" ]; then
    echo "✅ cuDNN header found"
else
    echo "❌ cuDNN header missing"
    echo "Please contact system administrator to install cuDNN:"
    echo "1. Download cuDNN v8.x for CUDA 12.9 from NVIDIA Developer portal"
    echo "2. Install using: sudo dpkg -i libcudnn8_*.deb"
fi

# Print TensorFlow configurations
echo -e "\n🔧 TensorFlow Configuration:"
echo "TF_CUDA_VERSION: $TF_CUDA_VERSION"
echo "TF_CUDNN_VERSION: $TF_CUDNN_VERSION"
echo "TF_CUDA_PATHS: $TF_CUDA_PATHS"