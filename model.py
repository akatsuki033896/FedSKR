# MLP
import torch
import torch.nn as nn
import numpy as np

# class RISphase(nn.Module):
#     def __init__(self, M):
#         super().__init__()
#         # 初始化相位
#         self.phi = nn.Parameter(torch.zeros(M)) # (M, )

#     # def forward(self, B):
#     #     return self.phi.unsqueeze(0).repeat(B, 1) # (B, M)

#     def get_phi(self):
#         return self.phi


class RISNet(nn.Module):
    def __init__(self, M, R, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * M * R, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, M)
        )

    def forward(self, H):
        # H: (B, M, R) complex

        H_real = H.real
        H_imag = H.imag

        # (B, M, R) → (B, 2MR)
        x = torch.cat([H_real, H_imag], dim=-1)
        x = x.reshape(x.shape[0], -1)

        phi = self.net(x)  # (B, M)

        return phi