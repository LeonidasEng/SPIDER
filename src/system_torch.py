import torch
import platform
import psutil

print("========== SPIDER SYSTEM CAPABILITY REPORT ==========")
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python Version: {platform.python_version()}")

# PyTorch + CUDA
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")

# CPU & RAM info
print(f"Logical CPU Cores: {psutil.cpu_count(logical=True)}")
print(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
print(f"Total RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")

print("======================================================")
