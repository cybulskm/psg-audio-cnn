#!/bin/bash

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Print current configuration
echo "Environment Setup:"
echo "----------------"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH includes CUDA: $(echo $PATH | grep -q cuda && echo 'Yes' || echo 'No')"
echo "LD_LIBRARY_PATH includes CUDA: $(echo $LD_LIBRARY_PATH | grep -q cuda && echo 'Yes' || echo 'No')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"