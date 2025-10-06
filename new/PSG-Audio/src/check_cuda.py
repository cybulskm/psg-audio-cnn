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
    
    # Check library paths
    lib_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu"
    ]
    
    for path in lib_paths:
        if os.path.exists(path):
            print(f"✓ Found library path: {path}")
        else:
            print(f"❌ Missing library path: {path}")
    
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
            else:
                print(f"❌ Missing {lib}")
        except Exception as e:
            print(f"❌ Error checking {lib}: {e}")

if __name__ == "__main__":
    check_cuda_setup()