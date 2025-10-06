#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.9"

python3 test_gpu.py