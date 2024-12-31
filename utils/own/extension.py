import psutil
import torch

def get_system_memory_usage():
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    return used_gb, total_gb, percent

def get_gpu_memory_usage():
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    total_gb = properties.total_memory / (1024**3)
    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
    max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    cached_gb = torch.cuda.memory_cached(device) / (1024**3)
    return total_gb, allocated_gb, max_allocated_gb, cached_gb

def print_memory_usage():
    system_used_memory, system_total_memory, system_memory_percent = get_system_memory_usage()
    total_gpu_memory, allocated_gpu_memory, max_allocated_gpu_memory, cached_gpu_memory = get_gpu_memory_usage()

#     print("=== CPU ===")
#     print(f"ToTal : {system_total_memory:.2f} GB")
#     print(f"Usage : {system_used_memory:.2f} GB")
#     print(f"Rate : {system_memory_percent:.2f}%")

    print("\n=== GPU  ===")
    print(f"total GPU memory: {total_gpu_memory:.2f} GB")
    print(f"allocated GPU memory: {allocated_gpu_memory:.2f} GB")
#     print(f"Max allocated GPU memory: {max_allocated_gpu_memory:.2f} GB")
    print(f"Cached GPU memory: {cached_gpu_memory:.2f} GB")