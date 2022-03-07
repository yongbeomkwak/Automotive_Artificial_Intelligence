import torch
import numpy as np


cpu_tensor = torch.zeros(2, 3)
device = torch.device("cpu")
tensor = cpu_tensor.to(device)
print(tensor)