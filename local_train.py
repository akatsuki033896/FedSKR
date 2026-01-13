import torch
import torch.nn as nn


def get_CSI_shape(H):
    # H.shape = (8000, 64, 4, 2)
    # 2:实部和虚部
    H = torch.tensor(H, dtype=torch.float32)
    H_complex = H[..., 0] + 1j * H[..., 1] # (8000, 64, 4) 恢复复数
    N, M, R = H_complex.shape # N：样本数
    return N, M, R, H_complex


def local_train(model, H, epochs, lr=1e-3, signSGD=False, batch_size=256):
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # N, M, R = H_complex.shape
    N, M, R, H_complex = get_CSI_shape(H)

    for _ in range(epochs):
        optimizer.zero_grad()
        idx = torch.randperm(N)[:batch_size]
        H_batch = H_complex[idx] # (B, M, R)


        # 最大化接收功率
        phi = model.phi # RIS每个单元的连续相位 (B,M)
        theta = torch.exp(1j * phi) # RIS反射系数 (B,M)
        h_eff = torch.sum(H_batch * theta.view(1, M, 1), dim=1) # (B, R)
        loss = -torch.mean(torch.abs(h_eff) ** 2) / M # 归一
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 梯度裁剪

        if signSGD:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr * p.grad.sign()
        else:
            optimizer.step()

    return model.state_dict(), loss.item() # type: ignore
