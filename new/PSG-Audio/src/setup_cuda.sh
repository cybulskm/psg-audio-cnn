#!/bin/bash

# Reset environment
unset LD_LIBRARY_PATH
unset CUDA_HOME

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# CUDA driver paths
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# TensorFlow specific settings for CUDA 12.x
export TF_CUDA_PATHS="/usr/local/cuda,/usr/local/cuda-12.9,/usr/lib/x86_64-linux-gnu"
export TF_CUDA_VERSION="12"
export TF_CUDNN_VERSION="9"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Print configuration
echo "Environment configured:"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "TF_CUDA_VERSION: $TF_CUDA_VERSION"
echo "TF_CUDNN_VERSION: $TF_CUDNN_VERSION"