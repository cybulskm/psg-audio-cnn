#!/bin/bash

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Version compatibility settings
export TF_CUDA_VERSION="12.9"
export TF_CUDNN_VERSION="9"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Run the test
echo "🔧 Environment configured:"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "-----------------"

python3 test_gpu.py