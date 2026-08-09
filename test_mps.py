import torch
from transformers import pipeline
print("MPS available:", torch.backends.mps.is_available())
