import torch
from utils import compute_skr_mimo

def ao_optimize_skr(H, H_e, max_iter=30, lr=0.1):
    M = H.shape[0]

    # ✅ 正确：leaf tensor
    phi = torch.nn.Parameter(
        torch.rand(M, device=H.device) * 2 * torch.pi - torch.pi
    )

    H = H.unsqueeze(0)
    H_e = H_e.unsqueeze(0)

    optimizer = torch.optim.Adam([phi], lr=lr)

    best_skr = -1
    best_theta = None

    for _ in range(max_iter):
        optimizer.zero_grad()

        theta = torch.exp(1j * phi).unsqueeze(0)

        skr = compute_skr_mimo(H, H_e, theta)
        loss = -skr.mean()

        loss.backward()
        optimizer.step()

        # ✅ 记录最好结果（防止震荡）
        if skr.item() > best_skr:
            best_skr = skr.item()
            best_theta = theta.detach().clone()

    return best_theta.squeeze(0)