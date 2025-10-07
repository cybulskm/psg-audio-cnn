import os
import sys
import tensorflow as tf
import subprocess
import platform
import glob

# Set environment variables for CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def check_cuda_driver():
    """Verify CUDA driver installation"""
    print("\n🔍 CUDA Driver Check:")
    print("-" * 50)
    
    # Check nvidia kernel module
    try:
        lsmod = subprocess.check_output('lsmod | grep nvidia', shell=True).decode()
        print("✅ NVIDIA kernel module loaded:")
        print(lsmod.strip())
    except:
        print("❌ NVIDIA kernel module not loaded")
    
    # Check driver files
    driver_paths = [
        '/usr/lib/x86_64-linux-gnu/libcuda.so',
        '/usr/lib/x86_64-linux-gnu/libnvidia-ml.so',
        '/dev/nvidia0'
    ]
    
    for path in driver_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")

def check_cuda_compatibility():
    """Check CUDA and TensorFlow version compatibility"""
    print("\n🔍 Version Compatibility Check:")
    print("-" * 50)
    
    # Get TensorFlow's CUDA requirements
    build_info = tf.sysconfig.get_build_info()
    tf_cuda = build_info.get('cuda_version', 'Unknown')
    tf_cudnn = build_info.get('cudnn_version', 'Unknown')
    
    print(f"TensorFlow {tf.__version__}:")
    print(f"- Requires CUDA: {tf_cuda}")
    print(f"- Requires cuDNN: {tf_cudnn}")
    
    # Get installed CUDA version
    try:
        nvcc = subprocess.check_output(['nvcc', '--version']).decode()
        cuda_version = nvcc.split('release ')[1].split(',')[0]
        print(f"\nInstalled CUDA: {cuda_version}")
        
        if cuda_version.startswith('12') and tf_cuda.startswith('11'):
            print("⚠️ Version mismatch: TensorFlow expects CUDA 11.x but CUDA 12.x is installed")
            print("Consider either:")
            print("1. Downgrading CUDA to version 11.8")
            print("2. Using tensorflow-nightly which supports CUDA 12.x")
    except:
        print("❌ Could not determine CUDA version")

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

def main():
    """Main verification routine"""
    check_cuda_driver()
    check_cuda_compatibility()
    check_environment()
    print_system_info()
    run_simple_gpu_test()

if __name__ == "__main__":
    main()