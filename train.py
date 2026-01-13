import torch
import os
import scipy.io as sio
import numpy as np
from model import RISphase
from local_train import local_train, get_CSI_shape
from CSI_process import split_data, normalization, sampling
from matplotlib import pyplot as plt


def fedavg(state_dicts):
    avg_dict = {}
    for key in state_dicts[0].keys():
        avg_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg_dict


def plot_loss(num_rounds, train_losses, client_num, local_epochs):
    # plot
    plt.plot(np.arange(num_rounds), train_losses, color='blue', label='train_losses')
    # plt.plot(np.arange(num_rounds), test_losses, color='red', label='test_losses')
    plt.xlabel('num_epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.title(f'client num = {client_num}')
    filename = f"R_{num_rounds}_C_{client_num}_LE_{local_epochs}.png"
    plt.savefig(filename)


# server(bS) train
def global_train(num_rounds, local_epochs, client_num, M):
    global_model = RISphase(M)
    avg_local_losses_sign = []

    for rnd in range(num_rounds):
        local_states = []
        local_losses = []

        for k in range(client_num):
            local_model = RISphase(M)
            local_model.load_state_dict(global_model.state_dict())

            state, loss = local_train(
                local_model,
                client_data[k],
                epochs=local_epochs,
                signSGD=True
            )

            local_states.append(state)
            local_losses.append(loss)

        # 聚合
        global_state = fedavg(local_states)
        global_model.load_state_dict(global_state)

        print(f"Round {rnd}: Avg local loss = {np.mean(local_losses):.4f}")

        avg_local_losses_sign.append(np.mean(local_losses))
    # print("local_epochs:", local_epochs)
    # print("losses:", avg_local_losses_sign)
    
    plot_loss(num_rounds, avg_local_losses_sign, client_num,local_epochs)
    
    return global_model.get_phi()


# 转化为np
def get_phi_numpy(phi):
    phi = phi.detach().numpy()
    # print(phi)
    np.save("optimized_phase.npy", phi)
    return


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # path = os.getcwd()
    # print(path)
    # file_path = os.path.join(current_dir, "RIS_Channels_MIMO.mat")
    file_path = os.path.join(current_dir, "RIS_CHannels1.mat")
    data = sio.loadmat(file_path)
    data = data['H'] # np.ndarray
    # print(data.shape)

    train_ratio = 0.8
    data_train, data_test = split_data(data, train_ratio)
    # plot_IQ_Distribution(data_train)

    data_train, data_test = normalization(data_train, data_test)
    # plot_IQ_Distribution(data_train)
    # plot_IQ_Distribution(data_test)

    H_real = data_train.real
    H_imag = data_train.imag
    H = np.stack([H_real, H_imag], axis=2) # 合并实部虚部
    # print(H.shape)
    
    H = np.transpose(H, (3, 0, 1, 2)) # 把实例数移到一维
    print(H.shape) # (8000, 64, 4, 2)
 
    #  parameters
    client_num = 40 # UE数量
    client_data = []
    client_data = sampling(H, client_num)
    if client_data == []:
        print("Invalid client number\n")
        exit(1)
    print(client_data)
    # print(type(client_data))
    # print(client_data[0])


    N, M, R, H_complex = get_CSI_shape(H)
    num_rounds = 50
    local_epochs = 40
    phi = global_train(num_rounds, local_epochs, client_num, M)
    phi = get_phi_numpy(phi)
    