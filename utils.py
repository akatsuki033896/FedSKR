import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 新增：LoS 生成（必须和 train.py 一致）=====
def generate_los(M, R):
    angle = torch.linspace(0, torch.pi, M, device=device)
    a = torch.exp(1j * angle)
    return a.unsqueeze(1).repeat(1, R)


def get_CSI_shape(H):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H = torch.tensor(H, dtype=torch.float32).to(device)
    H_complex = H[..., 0] + 1j * H[..., 1]
    N, M, R = H_complex.shape
    return N, M, R, H_complex


def plot_skr_optimizers(train_losses_dict, test_losses_dict=None, skr_random=None,
                        num_rounds=50, client_num=10, local_epochs=5):
    plt.figure(figsize=(8,5))
    rounds = np.arange(num_rounds)

    colors = {'Adam':"#e3d727", 'SignAdam':"#353535", 'SGD':'#2ca02c', 'SignSGD':"#9a4bce"}

    for opt_name, train_losses in train_losses_dict.items():
        plt.plot(rounds, [-l for l in train_losses],
                 color=colors[opt_name], linestyle='-', marker='o', markersize=4,
                 label=f'{opt_name} Train SKR')

    if test_losses_dict is not None:
        for opt_name, test_losses in test_losses_dict.items():
            plt.plot(rounds, [-l for l in test_losses],
                     color=colors[opt_name], linestyle='--', marker='s', markersize=4,
                     label=f'{opt_name} Test SKR')

    if skr_random is not None:
        plt.hlines(skr_random, 0, num_rounds-1, colors='gray', linestyles=':', label='Random SKR')

    plt.title(f'SKR Comparison of Optimizers (Clients={client_num}, Local Epochs={local_epochs})', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('SKR', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'SKR_optimizers_C{client_num}_LE{local_epochs}.png', dpi=300)
    plt.close()


def plot_loss(num_rounds, train_losses, test_losses=None, client_num=1, local_epochs=1, skr_random=None):
    """
    绘制训练和测试的 SKR 曲线，同时展示随机基线。
    
    train_losses/test_losses: 训练/测试的 -SKR.mean()
    skr_random: 随机 RIS 的 SKR 标量或列表
    """
    plt.figure(figsize=(8,5))
    rounds = np.arange(num_rounds)
    
    # 转换为 SKR
    train_skr = [-l for l in train_losses]
    plt.plot(rounds, train_skr, color='#1f77b4', linestyle='-', marker='o', markersize=4, label='Train SKR')
    
    if test_losses is not None:
        test_skr = [-l for l in test_losses]
        plt.plot(rounds, test_skr, color='#ff7f0e', linestyle='--', marker='s', markersize=4, label='Test SKR')
    
    # 随机基线
    if skr_random is not None:
        if np.isscalar(skr_random):
            plt.hlines(skr_random, 0, num_rounds-1, colors='gray', linestyles=':', label='Random SKR')
        else:
            plt.plot(rounds, skr_random, color='gray', linestyle=':', label='Random SKR')
    
    # 标题和标签
    plt.title(f'SKR Curves (Clients={client_num}, Local Epochs={local_epochs})', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('SKR', fontsize=12)
    
    # 网格和图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')
    
    # 坐标字体和布局
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    filename = f"R_{num_rounds}_C_{client_num}_LE_{local_epochs}_SKR.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 转化为.npy
# def get_phi_numpy(phi):
#     phi = phi.detach().numpy()
#     # print(phi)
#     np.save("optimized_phase.npy", phi)
#     return

def get_phi(model, H_sample):
    with torch.no_grad():
        phi = model(H_sample)
    return phi

# secret key rate
def compute_skr_mimo(H_batch, H_e_batch, theta, sigma2=0.1, sigma2_e=1.0):
    """
    MIMO版本 SKR(log-det) —— 支持 Adaptive RIS

    H_batch:   (B, M, R) 合法信道
    H_e_batch: (B, M, R) Eve信道
    theta:     (B, M) RIS相位 (complex)

    return:
        skr: (B,)
    """

    B, M, R = H_batch.shape

    # ===== RIS作用后等效信道 =====
    # (B, M, R) * (B, M, 1) -> (B, R)
    theta = theta.unsqueeze(-1)  # (B, M, 1)

    h_eff = torch.sum(H_batch * theta, dim=1)      # (B, R)
    h_eff_e = torch.sum(H_e_batch * theta, dim=1)  # (B, R)

    # ===== 构造 MIMO channel =====
    H_eff = h_eff.unsqueeze(1)      # (B, 1, R)
    H_eff_e = h_eff_e.unsqueeze(1)  # (B, 1, R)

    # ===== I 矩阵 =====
    I = torch.eye(R, dtype=torch.cfloat, device=H_batch.device).unsqueeze(0).repeat(B, 1, 1)

    # ===== 合法链路 log-det =====
    gram = H_eff.conj().transpose(1, 2) @ H_eff     # (B, R, R)
    mat = I + (1 / sigma2) * gram

    sign, logdet = torch.linalg.slogdet(mat)
    legit_rate = logdet / torch.log(torch.tensor(2.0, dtype=torch.float32, device=H_batch.device))

    # ===== Eve 链路 =====
    gram_e = H_eff_e.conj().transpose(1, 2) @ H_eff_e
    mat_e = I + (1 / sigma2_e) * gram_e

    sign_e, logdet_e = torch.linalg.slogdet(mat_e)
    eve_rate = logdet_e / torch.log(torch.tensor(2.0, dtype=torch.float32, device=H_batch.device))

    skr = legit_rate - eve_rate

    # ===== 非负约束 =====
    skr = torch.clamp(skr, min=0)

    return skr