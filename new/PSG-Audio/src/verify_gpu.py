import os
import sys
import tensorflow as tf
import subprocess
import platform

# Set environment variables for CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def check_environment():
    """Check CUDA environment setup"""
    print("\n🔧 Environment Check:")
    print("-" * 50)
    env_vars = {
        'CUDA_HOME': os.getenv('CUDA_HOME', 'Not Set'),
        'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', 'Not Set'),
        'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES', 'Not Set'),
        'PATH': os.getenv('PATH', 'Not Set')
    }
    
    for var, value in env_vars.items():
        print(f"{var}: {value}")

def print_system_info():
    """Print detailed system information"""
    print("🖥️ System Information:")
    print("-" * 50)
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Platform: {platform.platform()}")
    
    # Get CUDA and cuDNN info from TensorFlow
    build_info = tf.sysconfig.get_build_info()
    print(f"\nTensorFlow Build Information:")
    print(f"CUDA Version (Build): {build_info.get('cuda_version', 'Unknown')}")
    print(f"cuDNN Version (Build): {build_info.get('cudnn_version', 'Unknown')}")
    
    # Get runtime CUDA version
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        cuda_version = [line for line in nvidia_smi.split('\n') if "CUDA Version" in line][0]
        print(f"CUDA Version (Runtime): {cuda_version.split(':')[-1].strip()}")
    except:
        print("CUDA Runtime: Not detected")
    
    # Check cuDNN availability
    try:
        cudnn_output = subprocess.check_output("ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn", shell=True).decode()
        print("\ncuDNN Libraries Found:")
        for line in cudnn_output.splitlines():
            print(f"  {line.strip()}")
    except:
        print("\ncuDNN: Not detected in LD_LIBRARY_PATH")

def test_gpu_availability():
    """Test GPU availability and configuration"""
    print("\n🔍 GPU Availability Test:")
    print("-" * 50)
    
    try:
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU devices available")
            return False
        
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for: {gpu.name}")
            except RuntimeError as e:
                print(f"⚠️ Memory growth setting failed for {gpu.name}: {e}")
        
        print(f"\n✅ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.device_type}: {gpu.name}")
        
        return True
    except Exception as e:
        print(f"❌ Error during GPU configuration: {str(e)}")
        return False

def run_simple_gpu_test():
    """Run a simple GPU computation test"""
    print("\n🧪 Running Simple GPU Test:")
    print("-" * 50)
    
    if not test_gpu_availability():
        return
    
    try:
        # Create and run operations on GPU
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
    check_environment()
    print_system_info()
    run_simple_gpu_test()