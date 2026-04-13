import torch
import torch.nn as nn
from utils import compute_skr_mimo, get_CSI_shape, generate_los

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def local_train(model, H, epochs, lr=1e-3, optim="Adam", batch_size=256):
    model.train()
    model = model.to(device)

    if optim == "Adam" or optim == "SignAdam":
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    elif optim == "SGD" or optim == "SignSGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    H = torch.tensor(H, dtype=torch.float32).to(device)
    H_complex = H[..., 0] + 1j * H[..., 1]
    N, M, R = H_complex.shape

    for _ in range(epochs):
        optimizer.zero_grad()

        # 合法用户
        idx = torch.randperm(N, device=device)[:batch_size]
        H_batch = H_complex[idx]

        # 加 LoS
        H_batch = H_batch + 0.5 * generate_los(M, R)

        # Eve 
        alpha = 0.8

        idx_e = torch.randperm(N, device=device)[:batch_size]
        mask = idx_e == idx
        idx_e[mask] = (idx_e[mask] + 1) % N

        # 加噪
        noise = 0.2 * (torch.randn_like(H_batch) + 1j * torch.randn_like(H_batch))
        H_e_batch = alpha * (H_complex[idx_e] + noise)

        phi = model(H_batch)              # (B, M)
        theta = torch.exp(1j * phi)       # (B, M)
        skr = compute_skr_mimo(H_batch, H_e_batch, theta)

        loss = -skr.mean()

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        if optim == "signAdam" or optim == "signSGD":
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr * p.grad.sign()
        else:
            optimizer.step()

    return model.state_dict(), loss.item()
