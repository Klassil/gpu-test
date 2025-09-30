import torch
import tensorflow as tf
import time

print("==== PYTORCH TEST ====")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Тест: множення матриць на GPU
a = torch.rand((5000, 5000), device="cuda")
b = torch.rand((5000, 5000), device="cuda")

start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # дочекатися виконання
print("PyTorch matmul time:", time.time() - start, "сек")

print("\n==== TENSORFLOW TEST ====")
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Тест: множення матриць на GPU
with tf.device("/GPU:0"):
    a = tf.random.uniform((5000, 5000))
    b = tf.random.uniform((5000, 5000))
    start = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()  # викликати обчислення
    print("TensorFlow matmul time:", time.time() - start, "сек")
