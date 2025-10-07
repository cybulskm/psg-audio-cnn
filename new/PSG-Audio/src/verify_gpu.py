import os
import sys
import platform
import subprocess
import tensorflow as tf

# Set environment variables for CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def check_cuda_compatibility():
    """Check CUDA and TensorFlow version compatibility"""
    print("\n🔍 Version Compatibility Check:")
    print("-" * 50)
    
    # Get TensorFlow version and build info
    print(f"TensorFlow {tf.__version__}")
    
    # Check if CUDA is available
    print("\nCUDA Support:")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU devices visible: {len(tf.config.list_physical_devices('GPU'))}")
    
    # Get installed CUDA version
    try:
        nvcc = subprocess.check_output(['nvcc', '--version']).decode()
        cuda_version = nvcc.split('release ')[1].split(',')[0]
        print(f"\nInstalled CUDA: {cuda_version}")
    except:
        print("\n❌ Could not determine CUDA version")
    
    # Check cuDNN
    try:
        cudnn_check = subprocess.check_output('ldconfig -N -v $(sed "s/:/ /" <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn', shell=True).decode()
        print("\ncuDNN libraries found:")
        print(cudnn_check.strip())
    except:
        print("\n❌ cuDNN not found in LD_LIBRARY_PATH")

def check_cuda_driver():
    """Verify CUDA driver installation"""
    # ...existing code...

def check_environment():
    """Check environment variables"""
    # ...existing code...

def run_simple_gpu_test():
    """Run a simple GPU computation test"""
    print("\n🧪 Running Simple GPU Test:")
    print("-" * 50)
    
    try:
        # List physical devices
        physical_devices = tf.config.list_physical_devices('GPU')
        print(f"GPUs found: {len(physical_devices)}")
        
        if physical_devices:
            # Enable memory growth
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Run test computation
            with tf.device('/GPU:0'):
                print("\nTesting GPU computation...")
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                print(f"✅ Matrix multiplication successful, shape: {c.shape}")
        else:
            print("❌ No GPU devices available")
            
    except Exception as e:
        print(f"❌ GPU test failed: {str(e)}")

def main():
    """Main verification routine"""
    check_cuda_driver()
    check_cuda_compatibility()
    check_environment()
    run_simple_gpu_test()

if __name__ == "__main__":
    main()