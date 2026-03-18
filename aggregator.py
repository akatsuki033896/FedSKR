import torch

def fedavg(state_dicts):
    avg_dict = {}
    for key in state_dicts[0].keys():
        avg_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg_dict


def euclidean_distance(x, y):
    x_flat = torch.cat([x.view(-1) for x in x.values()])
    y_flat = torch.cat([y.view(-1) for y in y.values()])
    return torch.norm(x_flat - y_flat)


def krum(state_dicts, n_byzantine=0):
    n_clients = len(state_dicts)
    if n_clients <= n_byzantine + 1:
        return fedavg(state_dicts)
    
    dist_matrix = torch.zeros(n_clients, n_clients)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            dist = euclidean_distance(state_dicts[i], state_dicts[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    k = n_clients - n_byzantine - 2
    scores = []
    for i in range(n_clients):
        distances = dist_matrix[i]
        sorted_distances = torch.sort(distances)[0]
        score = torch.sum(sorted_distances[:k])
        scores.append(score)
    
    selected_idx = torch.argmin(torch.tensor(scores))
    return state_dicts[selected_idx]