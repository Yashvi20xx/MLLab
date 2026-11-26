
import torch
import time

print("========== MY AI LAB ==========")

print("Python is running PyTorch!")
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

print("\nChecking GPU...")

if torch.cuda.is_available():

    print("GPU AVAILABLE!")

    print("GPU Name:",
          torch.cuda.get_device_name(0))

    print("GPU Memory:",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
          "GB")

    print("\nStarting GPU calculations...")

    # Create two matrices directly in GPU memory
    a = torch.rand(4096, 4096, device="cuda")
    b = torch.rand(4096, 4096, device="cuda")

    start = time.time()

    # Repeat multiplication to keep GPU busy
    for i in range(30):
        result = torch.matmul(a, b)

    # Wait for GPU to finish
    torch.cuda.synchronize()

    end = time.time()

    print("\nGPU CALCULATION COMPLETED!")
    print("Time taken:", round(end - start, 3), "seconds")

    print("GPU memory currently allocated:",
          round(torch.cuda.memory_allocated() / 1024**2, 2),
          "MB")

else:
    print("CUDA NOT AVAILABLE!")
    print("We need to troubleshoot GPU access.")

print("\n========== END ==========")
