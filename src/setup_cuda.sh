#!/bin/bash

# Reset environment
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDNN_PATH

# Set CUDA and cuDNN paths
export CUDA_HOME=/usr/local/cuda
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu

# Set library paths in correct order
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Add CUDA binary path
export PATH=$CUDA_HOME/bin:$PATH

# NVIDIA runtime settings
export NVIDIA_VISIBLE_DEVICES="all"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_CACHE_DISABLE=0

# TensorFlow specific settings
export TF_CUDA_PATHS="$CUDNN_PATH,$CUDA_HOME,/usr/local/cuda-12.9"
export TF_CUDA_VERSION="12"
export TF_CUDNN_VERSION="9"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Print configuration
echo "🔧 Environment configured:"
echo "-----------------------"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_PATH: $CUDNN_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "TF_CUDA_PATHS: $TF_CUDA_PATHS"

# Verify library configuration
echo -e "\n📚 Library Status:"
echo "-----------------------"
if ldconfig -p | grep -q libcudnn; then
    echo "✅ cuDNN libraries found"
else
    echo "❌ cuDNN libraries not in cache"
fi

if ldconfig -p | grep -q libcuda; then
    echo "✅ CUDA libraries found"
else
    echo "❌ CUDA libraries not in cache"
fi