import time

import torch

gpu_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {gpu_device}")

# define size
size = 5000

# create random tensors on the device
mat_cpu1 = torch.rand(size, size)
mat_cpu2 = torch.rand(size, size)

# measure time
st = time.time()
res_cpu = torch.mm(mat_cpu1, mat_cpu2)
cpu_time = time.time() - st

print(f"Time on CPU: {cpu_time:.6f} seconds")

# moving tensors to GPU
mat_gpu1 = mat_cpu1.to(gpu_device)
mat_gpu2 = mat_cpu2.to(gpu_device)

# measure time
st = time.time()
res_gpu = torch.mm(mat_gpu1, mat_gpu2)
torch.cuda.synchronize()  # wait for GPU to finish
gpu_time = time.time() - st

print(f"Time on GPU: {gpu_time:.6f} seconds")

# compare results
print(f"\nSpeedup (CPU time / GPU time): {cpu_time / gpu_time}")
