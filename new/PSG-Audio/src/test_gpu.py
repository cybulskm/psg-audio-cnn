import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

print("🔍 Checking GPU availability for TensorFlow...")
print("-" * 50)

try:
    import tensorflow as tf
    
    # Check TensorFlow version
    print(f"✓ TensorFlow version: {tf.__version__}")
    
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
        
        # Test if CUDA is properly configured
        with tf.device('/GPU:0'):
            print("\n✨ Testing GPU computation...")
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("✓ GPU computation test successful!")
            
        print("\n✅ GPU acceleration is AVAILABLE and WORKING!")
    else:
        print("\n⚠️ No GPU devices detected - running on CPU only")
        print("If you have a GPU, check your CUDA and cuDNN installation")
        
except Exception as e:
    print(f"\n❌ Error checking GPU: {str(e)}")
    print("Please check your TensorFlow installation and CUDA configuration")