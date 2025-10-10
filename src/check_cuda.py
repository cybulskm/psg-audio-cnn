import os
import subprocess
import sys

def check_cuda_setup():
    """Verify CUDA installation and symlinks"""
    
    print("🔍 Checking CUDA Setup")
    print("-" * 50)
    
    # Check CUDA symlink
    cuda_path = "/usr/local/cuda"
    if os.path.islink(cuda_path):
        real_path = os.path.realpath(cuda_path)
        print(f"✓ CUDA symlink: {cuda_path} -> {real_path}")
    else:
        print("❌ CUDA symlink not found")
    
    # Check CUDA version
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        cuda_version = nvcc_output.split('release ')[1].split(',')[0]
        print(f"✓ CUDA Version: {cuda_version}")
    except:
        print("❌ Could not determine CUDA version")
    
    # Check TensorFlow version and compatibility
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        tf_cuda = tf.sysconfig.get_build_info()['cuda_version']
        tf_cudnn = tf.sysconfig.get_build_info()['cudnn_version']
        print(f"✓ TensorFlow {tf_version} (expects CUDA {tf_cuda}, cuDNN {tf_cudnn})")
    except:
        print("❌ Could not determine TensorFlow versions")
    
    # Check library paths
    lib_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu"
    ]
    
    print("\nChecking Library Paths:")
    for path in lib_paths:
        if os.path.exists(path):
            print(f"✓ Found path: {path}")
            # List .so files in this path
            try:
                so_files = [f for f in os.listdir(path) if f.endswith('.so')]
                if 'cudnn' in ' '.join(so_files):
                    print(f"  └─ Found cuDNN in this path")
            except:
                pass
        else:
            print(f"❌ Missing path: {path}")
    
    # Check critical CUDA libraries
    libraries = [
        "libcudart.so",
        "libcublas.so",
        "libcudnn.so"
    ]
    
    print("\nChecking CUDA Libraries:")
    for lib in libraries:
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"✓ Found {lib}")
                # Try to get the actual path
                lib_lines = [l for l in result.stdout.split('\n') if lib in l]
                if lib_lines:
                    print(f"  └─ {lib_lines[0].split(' => ')[1]}")
            else:
                print(f"❌ Missing {lib}")
        except Exception as e:
            print(f"❌ Error checking {lib}: {e}")

    print("\n🔍 Recommendation:")
    print("Based on your CUDA 12.9 installation:")
    print("1. Install tensorflow-gpu==2.15.0 (supports CUDA 12.x)")
    print("2. Install cuDNN library (required for TensorFlow)")
    print("3. Update LD_LIBRARY_PATH if needed")

if __name__ == "__main__":
    check_cuda_setup()