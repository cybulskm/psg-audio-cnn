import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import sys
import tensorflow as tf
import subprocess
import platform

def print_system_info():
    print("🖥️ System Information:")
    print("-" * 50)
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Platform: {platform.platform()}")
    
    # Get CUDA version from nvidia-smi
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        cuda_version = [line for line in nvidia_smi.split('\n') if "CUDA Version" in line][0]
        print(f"CUDA (nvidia-smi): {cuda_version.split(':')[-1].strip()}")
    except:
        print("CUDA: Not found in nvidia-smi")

def test_gpu_availability():
    print("\n🔍 GPU Availability Test:")
    print("-" * 50)
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU devices available")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.device_type}: {gpu.name}")
    
    return True

def run_simple_gpu_test():
    print("\n🧪 Running Simple GPU Test:")
    print("-" * 50)
    
    if not test_gpu_availability():
        return
    
    try:
        # Create a simple operation on GPU
        with tf.device('/GPU:0'):
            print("Creating test tensors...")
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            print("Performing matrix multiplication...")
            c = tf.matmul(a, b)
            print(f"✅ Result shape: {c.shape}")
            print("GPU test successful!")
    except Exception as e:
        print(f"❌ GPU test failed: {str(e)}")

if __name__ == "__main__":
    print_system_info()
    run_simple_gpu_test()