import os
import subprocess
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

def check_nvidia_smi():
    """Check NVIDIA GPU status using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvidia-smi not found. NVIDIA driver may not be installed."

def check_cuda_version():
    """Check CUDA version from nvcc if available"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvcc not found. CUDA toolkit may not be installed."

print("🔍 GPU Environment Diagnostic")
print("-" * 50)

# 1. Check System Environment
print("\n1️⃣ System Environment:")
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")

# 2. Check NVIDIA Driver and GPU
print("\n2️⃣ NVIDIA GPU Status:")
print(check_nvidia_smi())

# 3. Check CUDA Installation
print("\n3️⃣ CUDA Installation:")
print(check_cuda_version())

# 4. Check TensorFlow GPU Support
print("\n4️⃣ TensorFlow GPU Support:")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Get build information
    build_info = tf.sysconfig.get_build_info()
    print(f"Build with CUDA: {build_info.get('is_cuda_build', False)}")
    print(f"CUDA Version: {build_info.get('cuda_version', 'Unknown')}")
    print(f"cuDNN Version: {build_info.get('cudnn_version', 'Unknown')}")
    
    # List physical devices
    physical_devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    print(f"\n📊 Device Information:")
    print(f"Total devices detected: {len(physical_devices)}")
    print(f"GPU devices detected: {len(gpu_devices)}")
    
    if gpu_devices:
        print("\n🎯 GPU Details:")
        for i, gpu in enumerate(gpu_devices):
            print(f"GPU {i}: {gpu.device_type} - {gpu.name}")
            
        # Try enabling memory growth
        print("\nAttempting to configure GPU memory growth...")
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ Memory growth enabled")
        except Exception as e:
            print(f"⚠️ Could not configure memory growth: {e}")
        
        # Test GPU computation
        with tf.device('/GPU:0'):
            print("\n✨ Testing GPU computation...")
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("✓ GPU computation test successful!")
            
        print("\n✅ GPU acceleration is AVAILABLE and WORKING!")
    else:
        print("\n⚠️ No GPU devices detected - running on CPU only")
        print("Possible reasons:")
        print("1. CUDA driver not installed or outdated")
        print("2. CUDA toolkit version mismatch with TensorFlow")
        print("3. GPU is being used by another process")
        print("4. Environment variables not set correctly")
        
except Exception as e:
    print(f"\n❌ Error checking TensorFlow GPU support: {str(e)}")
    print("Please check:")
    print("1. TensorFlow installation (pip list | grep tensorflow)")
    print("2. CUDA toolkit installation (nvcc --version)")
    print("3. NVIDIA driver installation (nvidia-smi)")
    print("4. Environment variables (CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH)")