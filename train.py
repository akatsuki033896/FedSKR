import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import scipy.io as sio
import numpy as np
from model import RISNet
from local_train import local_train, get_CSI_shape, device
from CSI_process import split_data, normalization, sampling
from utils import plot_loss, compute_skr_mimo, generate_los, plot_skr_optimizers
from aggregator import fedavg, krum

def evaluate_on_data(model, H, batch_size=256, fixed_idx=None):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        H = torch.tensor(H, dtype=torch.float32).to(device)
        H_complex = H[..., 0] + 1j * H[..., 1]
        N, M, R = H_complex.shape

        if fixed_idx is None:
            idx = torch.arange(min(batch_size, N), device=device)
        else:
            idx = fixed_idx

        H_batch = H_complex[idx]

        # ===== 加 LoS（合法用户更强）=====
        H_batch = H_batch + 0.5 * generate_los(M, R)

        # ===== Eve：更弱 + 扰动 =====
        alpha = 0.8
        idx_e = (idx + 1) % N
        
        # noise = 0.2 * (torch.randn_like(H_batch) + 1j * torch.randn_like(H_batch))
        noise = 0
        H_e_batch = alpha * (H_complex[idx_e] + noise)

        # phi = model.phi
        # theta = torch.exp(1j * phi)
        phi = model(H_batch)          # (B, M)
        theta = torch.exp(1j * phi)   # (B, M)

        skr = compute_skr_mimo(H_batch, H_e_batch, theta)
        loss = -skr.mean()

    return loss.item()


# ===== server train =====
def global_train(num_rounds, local_epochs, client_num, M, R, test_data=None,
                 aggregator='fedavg', n_byzantine=0, optim="Adam"):
    global_model = RISNet(M, R).to(device)
    avg_local_losses = []
    avg_test_losses = []

    fixed_idx = torch.arange(256, device=device)

    # 基线
    if test_data is not None:
        H_eval = torch.tensor(test_data, dtype=torch.float32).to(device)
        H_eval_complex = H_eval[..., 0] + 1j * H_eval[..., 1]
        N, _, _ = H_eval_complex.shape

        H_batch = H_eval_complex[fixed_idx]
        H_batch = H_batch + 0.5 * generate_los(M, H_batch.shape[2])

        idx_e = (fixed_idx + 1) % N
        H_e_batch = 0.8 * (H_eval_complex[idx_e] + 0)

        phi_random = torch.rand(M, device=device) * 2 * np.pi
        theta_random = torch.exp(1j * phi_random)

        skr_random = compute_skr_mimo(H_batch, H_e_batch, theta_random)
        skr_random_mean = skr_random.mean().item()
        print("\n===== Baseline =====")
        print(f"Random RIS SKR: {skr_random_mean:.4f}\n")
    else:
        skr_random_mean = None

    # 训练
    for rnd in range(num_rounds):
        local_states = []
        local_losses = []

        for k in range(client_num):
            local_model = RISNet(M, R).to(device)
            local_model.load_state_dict(global_model.state_dict())

            state, loss = local_train(
                local_model,
                client_data[k],
                epochs=local_epochs,
                optim=optim
            )

            local_states.append(state)
            local_losses.append(loss)

        # 聚合
        if aggregator == 'krum':
            global_state = krum(local_states, n_byzantine)
        else:
            global_state = fedavg(local_states)

        global_model.load_state_dict(global_state, strict=False)

        avg_loss = np.mean(local_losses)
        avg_local_losses.append(avg_loss)

        print(f"Round {rnd}: Avg SKR = {-avg_loss:.4f}")

        # 测试
        if test_data is not None:
            test_loss = evaluate_on_data(global_model, test_data, fixed_idx=fixed_idx)
            avg_test_losses.append(test_loss)
            print(f"Round {rnd}: Test SKR = {-test_loss:.4f}")

    # 最终结果
    if test_data is not None:
        final_loss = evaluate_on_data(global_model, test_data, fixed_idx=fixed_idx)
        print("\n===== Final Result =====")
        print(f"Final SKR: {-final_loss:.4f}")
        print(f"Improvement over Random: {-final_loss - skr_random_mean:.4f}")

    return global_model.state_dict(), avg_local_losses, avg_test_losses, skr_random_mean


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "RIS_Channels.mat")

    data = sio.loadmat(file_path)
    data = data['H']

    train_ratio = 0.8
    data_train, data_test = split_data(data, train_ratio)

    data_train, data_test = normalization(data_train, data_test)

    # ===== train =====
    H_real = data_train.real
    H_imag = data_train.imag
    H = np.stack([H_real, H_imag], axis=2)
    H = np.transpose(H, (3, 0, 1, 2))
    print("Train shape:", H.shape)

    # ===== test =====
    H_test_real = data_test.real
    H_test_imag = data_test.imag
    H_test = np.stack([H_test_real, H_test_imag], axis=2)
    H_test = np.transpose(H_test, (3, 0, 1, 2))
    print("Test shape:", H_test.shape)

    # ===== clients =====
    client_num = 10
    client_data = sampling(H, client_num)

    if client_data == []:
        print("Invalid client number\n")
        exit(1)

    N, M, R, H_complex = get_CSI_shape(H)

    num_rounds = 50
    local_epochs = 40

    optimizers = ["Adam", "SignAdam", "SGD", "SignSGD"]
    train_losses_dict = {}
    test_losses_dict = {}

    for opt in optimizers:
        _, train_losses, test_losses, skr_random_mean = global_train(
            num_rounds, local_epochs, client_num, M, R,
            test_data=H_test, aggregator='krum', n_byzantine=5, optim=opt
        )
        train_losses_dict[opt] = train_losses
        test_losses_dict[opt] = test_losses

    # 绘制四种优化器对比图
    plot_skr_optimizers(train_losses_dict, test_losses_dict, skr_random_mean,
                        num_rounds=num_rounds, client_num=client_num, local_epochs=local_epochs)