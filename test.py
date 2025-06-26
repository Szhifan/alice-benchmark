import torch 
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

param = nn.Parameter(torch.randn(4), requires_grad=True)
print(param)