import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import os


def split_data(data, train_ratio):
    N = data.shape[2] # type: ignore
    idx = np.random.permutation(N)
    N_train = int(train_ratio * N)

    train_idx = idx[:N_train]
    test_idx = idx[N_train:]

    data_train = data[:, :, train_idx] # (64, 4, 8000)
    data_test = data[:, :, test_idx] # (64, 4, 2000)

    train_num = data_train.shape[2]
    val_num = data_test.shape[2]
    return data_train, data_test


def plot_IQ_Distribution(data):
    plt.scatter(
        data[:, :, :].real.flatten(),
        data[:, :, :].imag.flatten(),
        s=1,
        alpha=0.3
    )
    plt.xlabel("Real Part")
    plt.ylabel("Imag Part")
    plt.title("IQ Distribution of RIS-MIMO Channel")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("IQ_Distribution")
    # plt.show()


# normalization
def z_score(x, mean_real, std_real, mean_imag, std_imag):
    x_real = (x.real - mean_real) / std_real
    x_imag = (x.imag - mean_imag) / std_imag
    return x_real + x_imag * 1j


def normalization(data_train, data_test):
    mean_real = np.mean(data_train.real)
    std_real = np.std(data_train.real) + 1e-8
    mean_imag = np.mean(data_train.imag)
    std_imag = np.std(data_train.imag) + 1e-8

    data_train = z_score(data_train, mean_real, std_real, mean_imag, std_imag)
    data_test = z_score(data_test, mean_real, std_real, mean_imag, std_imag)
    return data_train, data_test


# 划分数据集给UE
# IID
# Non-IID: shuffle=False
def sampling(H, client_num, shuffle=True, seed=0):
    sample_sum = H.shape[0]
    
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(sample_sum)
        H = H[idx]

    # 样本数不能整除UE数 返回空list
    if sample_sum % client_num != 0:
        return []
    
    samples_per_client = sample_sum // client_num
    
    client_data = []
    for k in range(client_num):
        x_k = H[k * samples_per_client:(k + 1) * samples_per_client]
        client_data.append(x_k)
    
    return client_data