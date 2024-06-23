import torch

"""
[Torch 실습 환경 확인]
- 아래 코드는 Torch 버전, GPU 가속 등의 실습 환경을 확인하는 코드입니다.
"""

# torch 버전 확인
print("torch 버전 확인")
print(torch.__version__)
print("")

# CUDA 가속 확인
print("CUDA 가속 확인")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
print(torch.version.cuda)
print("")

# MPS 가속 확인
print("MPS 가속 확인")
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
print("")
