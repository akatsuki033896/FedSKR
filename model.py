# MLP
import torch
import torch.nn as nn
import numpy as np

# supervised test
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 4 * 2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         x = x.reshape(x.size(0), -1)
#         return self.fc(x).squeeze()


class RISphase(nn.Module):
    def __init__(self, M):
        super().__init__()
        # 初始化相位
        self.phi = nn.Parameter(torch.zeros(M)) # (M, )

    # def forward(self, B):
    #     return self.phi.unsqueeze(0).repeat(B, 1) # (B, M)

    def get_phi(self):
        return self.phi