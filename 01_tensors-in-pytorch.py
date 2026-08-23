import sys

import numpy as np
import torch

print("=== CPU & SYSTEM INFO ===")
# Get your Python and PyTorch installation versions
print(f"OS Platform: {sys.platform}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
# Note: PyTorch runs on the CPU by default
print(f"CPU Threads Available: {torch.get_num_threads()}")

print("\n=== GPU & HARDWARE ACCELERATION INFO ===")
# Dynamically select the best available accelerator: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    cuda_mem_allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
    cuda_mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 2)

    print(f"Active Acceleration: NVIDIA CUDA ({gpu_name})")
    print(f"CUDA Device Index: {current_device}")
    print(f"Available CUDA Devices: {torch.cuda.device_count()}")
    print(f"PyTorch Allocated GPU Memory: {cuda_mem_allocated:.2f} MB")
    print(f"PyTorch Reserved Driver Memory: {cuda_mem_reserved:.2f} MB")

elif torch.backends.mps.is_available():
    device = torch.device("mps")
    current_mem = torch.mps.current_allocated_memory() / (1024 ** 2)
    driver_mem = torch.mps.driver_allocated_memory() / (1024 ** 2)

    print("Active Acceleration: Apple Silicon Integrated Matrix/GPU (MPS)")
    print(f"PyTorch Allocated GPU Memory: {current_mem:.2f} MB")
    print(f"Total Framework Allocated Driver Memory: {driver_mem:.2f} MB")

else:
    device = torch.device("cpu")
    print("GPU acceleration not available. Running in CPU-only mode.")

print(f"Default Selected Target Device: {device}")


print("\n=== BASIC TENSOR CREATION ===")
# empty tensor
ten1: torch.Tensor = torch.empty((2, 3))
print(f"\nEmpty Tensor (2x3):\n{ten1}")
print(f"type: {type(ten1)}, shape: {ten1.shape}, dtype: {ten1.dtype}")

# zero and ones tensor
ten2: torch.Tensor = torch.zeros((2, 3))
print(f"\nZeros Tensor (2x3):\n{ten2}")
ten3: torch.Tensor = torch.ones((2, 3))
print(f"\nOnes Tensor (2x3):\n{ten3}")

# random tensor (with or without a seed)
torch.manual_seed(42)  # Set a seed for reproducibility
ten4: torch.Tensor = torch.rand((2, 3))
print(f"\nRandom Tensor (2x3):\n{ten4}")


print("\n=== MORE WAYS TO CREATE TENSORS ===")
# from existing Python data (list / nested list)
ten5: torch.Tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\nFrom nested list:\n{ten5}")

# from a NumPy array (shares memory with the array on CPU!)
np_array = np.array([1.0, 2.0, 3.0])
ten6: torch.Tensor = torch.from_numpy(np_array)
print(f"\nFrom NumPy array:\n{ten6}")
np_array[0] = 99.0  # mutate original array
print(f"After mutating source array, tensor also changes (shared memory): {ten6}")

# ranges and evenly-spaced values
ten7: torch.Tensor = torch.arange(0, 10, 2)          # like Python's range()
ten8: torch.Tensor = torch.linspace(0, 1, steps=5)   # evenly spaced floats
print(f"\narange(0, 10, 2): {ten7}")
print(f"linspace(0, 1, steps=5): {ten8}")

# identity / diagonal matrix
ten9: torch.Tensor = torch.eye(3)
print(f"\nIdentity Matrix (3x3):\n{ten9}")

# fill an arbitrary shape with a constant value
ten10: torch.Tensor = torch.full((2, 2), fill_value=7)
print(f"\nFull Tensor (2x2) filled with 7:\n{ten10}")

# "_like" variants: copy the shape/dtype/device of an existing tensor
ten11: torch.Tensor = torch.zeros_like(ten5)
ten12: torch.Tensor = torch.ones_like(ten5)
ten13: torch.Tensor = torch.rand_like(ten5, dtype=torch.float32)
print(f"\nzeros_like(ten5):\n{ten11}")
print(f"ones_like(ten5):\n{ten12}")
print(f"rand_like(ten5, dtype=float32):\n{ten13}")

# random integers in a range
ten14: torch.Tensor = torch.randint(low=0, high=10, size=(2, 3))
print(f"\nRandom Integers [0, 10) (2x3):\n{ten14}")

# normal (Gaussian) distribution, mean=0, std=1
ten15: torch.Tensor = torch.randn((2, 3))
print(f"\nRandom Normal Tensor (2x3):\n{ten15}")

# specifying dtype and device dynamically (runs safely on CUDA, MPS, or CPU)
ten16: torch.Tensor = torch.ones((2, 2), dtype=torch.float32, device=device)
print(f"\nExplicit dtype/device Tensor:\n{ten16}, dtype: {ten16.dtype}, device: {ten16.device}")


print("\n=== ELEMENT-WISE OPERATIONS ===")
a: torch.Tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b: torch.Tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"a:\n{a}\nb:\n{b}")

# arithmetic: operator overloads and their torch.* equivalents
print(f"\na + b (add):\n{a + b}")
print(f"a - b (sub):\n{a - b}")
print(f"a * b (element-wise mul):\n{a * b}")   # NOT matrix multiplication!
print(f"a / b (div):\n{a / b}")
print(f"a ** 2 (power):\n{a ** 2}")

# in-place variants mutate the tensor and are marked with a trailing "_"
c: torch.Tensor = a.clone()
c.add_(b)  # equivalent to: c = c + b, without allocating a new tensor
print(f"\nc.add_(b) (in-place):\n{c}")

# common unary math functions
print(f"\nsqrt(a): {torch.sqrt(a)}")
print(f"exp(a): {torch.exp(a)}")
print(f"log(a): {torch.log(a)}")

# comparisons produce a boolean tensor
print(f"\na > 2: {a > 2}")

# broadcasting
row: torch.Tensor = torch.tensor([10.0, 20.0])
print(f"\na (2x2) + row (2,) via broadcasting:\n{a + row}")


print("\n=== MATRIX OPERATIONS ===")
m1: torch.Tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
m2: torch.Tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"m1:\n{m1}\nm2:\n{m2}")

print(f"\nm1 @ m2 (matmul via operator):\n{m1 @ m2}")
print(f"torch.matmul(m1, m2):\n{torch.matmul(m1, m2)}")
print(f"torch.mm(m1, m2) (2D-only matmul):\n{torch.mm(m1, m2)}")

# batched matrix multiplication
batch1: torch.Tensor = torch.rand((4, 2, 3))
batch2: torch.Tensor = torch.rand((4, 3, 2))
batch_result: torch.Tensor = torch.bmm(batch1, batch2)
print(f"\ntorch.bmm on batch of 4 (2x3)@(3x2) matrices -> shape: {batch_result.shape}")

# transpose
print(f"\nm1.T (transpose):\n{m1.T}")

# dot product of two 1D vectors
v1: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
v2: torch.Tensor = torch.tensor([4.0, 5.0, 6.0])
print(f"\ntorch.dot(v1, v2): {torch.dot(v1, v2)}")

# determinant and inverse
print(f"\ntorch.linalg.det(m1): {torch.linalg.det(m1)}")
print(f"torch.linalg.inv(m1):\n{torch.linalg.inv(m1)}")
