import os

import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.decomposition import NMF
import torch
from utils.get_centrality import centralissimo
def to_device(x, device):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64)
    return x.to(device)
def row_normalize(X):
    return X / (torch.sum(X, dim=1, keepdim=True) + 1e-8)


def update_G(G, A, H, U, W, beta, momentum=0.5):
    R = A - G @ H
    numer = R @ H.T + beta * (U @ W.T)
    denom = torch.sum(H * H, dim=1) + beta  # shape (K,)
    G_new = numer / (denom[None, :] + 1e-8)
    # 缓冲更新，减缓震荡
    G = momentum * G + (1 - momentum) * G_new
    G = torch.clamp(G, 1e-8, 1e6)
    return G


# 向量化版
def update_U(U, C, V, G, W, beta, gamma):
    CV = C - U @ V
    numer1 = beta * (CV @ V.T)
    GW = G - U @ W.T
    numer2 = gamma * (GW @ W)
    denom = beta * torch.sum(V * V, dim=1) + gamma * torch.sum(W * W, dim=0)
    U = (numer1 + numer2) / (denom + 1e-8)
    U = torch.clamp(U, 1e-8, 1e6)
    return U

def update_H(A, G,H):
    R = A - G @ H
    numer = G.T @ R
    denom = torch.sum(G * G, dim=0)[:, None]
    H_new = numer / (denom + 1e-8)
    H_new = torch.clamp(H_new, -1e3, 1e3)
    H = torch.clamp(H, 1e-8, 1e6)
    return H_new

# 辅助函数

def update_U(U, C, V, G, W, beta, gamma, momentum=0.5):
    CV = C - U @ V
    GW = G - U @ W.T
    numer = beta * (CV @ V.T) + gamma * (GW @ W)
    denom = beta * torch.sum(V * V, dim=1) + gamma * torch.sum(W * W, dim=0)
    U_new = numer / (denom[None, :] + 1e-8)
    # 缓冲更新
    U = momentum * U + (1 - momentum) * U_new
    U = torch.clamp(U, -1e3, 1e3)
    U= torch.clamp(U, 1e-8, 1e6)
    return U

def update_V(C, V, U):
    R = C - U @ V
    numer = U.T @ R
    denom = torch.sum(U * U, dim=0)[:, None]
    V_new = numer / (denom + 1e-8)
    V_new= torch.clamp(V_new, 1e-8, 1e6)
    return V_new

def update_W(U, G):
    svd_matrix = G.T @ U
    Q, _ = torch.linalg.qr(svd_matrix)
    W = Q

    W = torch.clamp(W, 1e-8, 1e6)
    return W


def calculate_loss(A, C, G, H, U, V, W, alpha, beta):
    """计算总损失函数值"""
    # 结构损失
    struct_loss = torch.sum((A - G @ H) ** 2)
    attr_loss   = torch.sum((C - U @ V) ** 2)
    dis_loss    = torch.sum((G - U @ W.T) ** 2)
    total_loss  = struct_loss + alpha * attr_loss + beta * dis_loss
    return total_loss


def one_embedding(A, C, K_iter,NoComm,beta=1.0,gamma=1.0, alpha=1.0, max_iter=100, tol=1e-4,device=None):


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = to_device(A, device)
    C = to_device(C, device)




    n_nodes, n_attrs = C.shape
    # NoComm代表数据集中不同类别的数量，也就是真实标签集合里不同元素的数量。K表示非负矩阵分解后矩阵的维度数量
    K = K_iter * NoComm   #Nocomm 等于数据集label种类数
    # 初始化
    model_A = NMF(
        n_components=K,
        init='nndsvda',  # 更稳定的初始化方式
        random_state=0,
        solver='mu',
        beta_loss='kullback-leibler',
        max_iter=500,  # 增加迭代次数

    )
    print("model A")
    G = model_A.fit_transform(A.cpu().numpy())
    H = model_A.components_

    # --- 分解 C ---
    model_C = NMF(
        n_components=K,
        init='nndsvda',  # 更稳定的初始化方式
        random_state=0,
        solver='mu',
        beta_loss='kullback-leibler',
        max_iter=500,  # 增加迭代次数

    )
    print("model C")
    U = model_C.fit_transform(C.cpu().numpy())
    V = model_C.components_

    # W = np.eye(K)
    U=to_device(U, device)
    V=to_device(V, device)
    G=to_device(G, device)
    H=to_device(H, device)


    W = torch.eye(K, device=device, dtype=torch.float64)
    prev_loss = float('inf')
    # 初始归一化
    G /= torch.norm(G) + 1e-8
    U /= torch.norm(U) + 1e-8
    H /= torch.norm(H) + 1e-8
    V /= torch.norm(V) + 1e-8
    for iter in range(max_iter):
        # 迭代更新
        G = update_G(G, A, H, U, W, beta)
        H = update_H(A, G,H)
        U = update_U(U, C, V, G, W, beta, gamma)
        V = update_V(C, V, U)
        W = update_W(U, G)



        # 计算损失
        current_loss = calculate_loss(A, C, G, H, U, V, W, alpha, beta)
        # 防止负数和爆炸


        # 检查收敛性
        loss_diff = abs(prev_loss - current_loss)
        if iter > 0 and loss_diff < tol:
            print(f"Converged at iteration {iter} with loss: {current_loss}")
            break

        prev_loss = current_loss


        print(f"Iteration {iter}, Loss: {current_loss}")
        LOSS=current_loss


    return G, H, U, V, W,LOSS

def run_one_embedding_safe(A_np, C_np, K_iter, NoComm,
                           alpha=1.0, beta=1.0,
                           max_iter=60, tol=1e-2,
                           device=None):
    """
    GPU 安全版 SemiONE embedding 调用
    A_np: numpy 邻接矩阵
    C_np: numpy 特征矩阵
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转成 float64，提升稳定性
    A = torch.tensor(A_np, dtype=torch.float64, device=device)
    C = torch.tensor(C_np, dtype=torch.float64, device=device)

    # 邻接矩阵归一化，避免数值爆炸
    D = torch.diag(1.0 / torch.sqrt(A.sum(dim=1) + 1e-10))
    A_norm = D @ A @ D

    # 调用原 one_embedding
    G, H, U, V, W, loss = one_embedding(
        A_norm, C, K_iter, NoComm,
        alpha=alpha, beta=beta,
        max_iter=max_iter, tol=tol, device=device
    )

    # GPU 安全 clamp，避免 loss inf
    clamp_val = 1e2
    G = torch.clamp(G, -clamp_val, clamp_val)
    H = torch.clamp(H, -clamp_val, clamp_val)
    U = torch.clamp(U, -clamp_val, clamp_val)
    V = torch.clamp(V, -clamp_val, clamp_val)
    W = torch.clamp(W, -clamp_val, clamp_val)

    # 再次检查 loss 是否正常
    if torch.isinf(loss) or torch.isnan(loss):
        print("[WARN] Loss is inf/nan after clamp! Consider reducing learning rate or clamp_val.")

    return G, H, U, V, W, loss


# 事后计算异常得分
def compute_outlier_scores(A, C, G, H, U, V, W, sigma1=1.0, sigma2=1.0, sigma3=1.0,device=None):
    # n_nodes = A.shape[0]
    # O1 = np.zeros(n_nodes)  # 结构异常得分
    # O2 = np.zeros(n_nodes)  # 属性异常得分
    # O3 = np.zeros(n_nodes)  # 不一致异常得分
    #
    # for i in range(n_nodes):
    #     # 计算结构重构误差
    #     struct_error = np.sum((A[i] - G[i] @ H) ** 2)
    #     O1[i] = np.exp(-struct_error / (2 * sigma1 ** 2))
    #
    #     # 计算属性重构误差
    #     attr_error = np.sum((C[i] - U[i] @ V) ** 2)
    #     O2[i] = np.exp(-attr_error / (2 * sigma2 ** 2))
    #
    #     # 计算不一致误差
    #     dis_error = np.linalg.norm(G[i] - U[i] @ W) ** 2
    #     O3[i] = np.exp(-dis_error / (2 * sigma3 ** 2))
    struct_error = torch.sum((A - G @ H) ** 2, dim=1)
    attr_error = torch.sum((C - U @ V) ** 2, dim=1)
    dis_error = torch.sum((G - U @ W.T) ** 2, dim=1)

    O1 = torch.exp(-struct_error / (2 * sigma1 ** 2))
    O2 = torch.exp(-attr_error / (2 * sigma2 ** 2))
    O3 = torch.exp(-dis_error / (2 * sigma3 ** 2))
    return O1, O2, O3


def compute_composite_outlier_score(O1, O2, O3, weights):
    """计算综合异常得分"""
    w1, w2, w3 = weights
    print(w1)
    print(w2)
    print(w3)
    return (w1 * O1 + w2 * O2 + w3 * O3) / (w1 + w2 + w3)

def extract_high_loss_nodes(O1, O2, O3,weights, one_budget,method='composite',device=None):
    """
    提取高损失节点

    参数:
    O1: 结构异常得分 (n_nodes,)
    O2: 属性异常得分 (n_nodes,)
    O3: 不一致异常得分 (n_nodes,)

    method: 'composite'（综合得分）、'structural'（结构）、'attribute'（属性）、'discrepancy'（不一致）

    返回:
    high_loss_indices: 高损失节点索引列表
    """

    O1 = np.array(O1).flatten()
    O2 = np.array(O2).flatten()
    O3 = np.array(O3).flatten()

    if method == 'composite':
        # 计算综合得分
        composite_score = compute_composite_outlier_score(O1, O2, O3,weights)
        score = composite_score
    elif method == 'structural':
        score = O1
    elif method == 'attribute':
        score = O2
    elif method == 'discrepancy':
        score = O3
    else:
        raise ValueError("Method must be 'composite', 'structural', 'attribute', or 'discrepancy'")
    # print(normcen.shape)
    print(score.shape)
    composite_score1 = score
    # threshold_per1=1-threshold_per
    # threshold = np.quantile(composite_score1, threshold_per1)
    # one_budget
    # 提取高损失节点（得分越高，损失越高）
    # high_loss_mask = composite_score1 > threshold
    # high_loss_node_id = np.where(high_loss_mask)[0]+1   #索引从0开始，index从1开始，所以要加1为节点索引
    # high_loss_node_id = np.where(high_loss_mask)[0]  #从0开始

    # 只取前 one_budget 个最大值的索引（比完整排序快） 复杂度：O(n)
    high_loss_node_id = np.argpartition(-composite_score1, one_budget)[:one_budget]



    return high_loss_node_id
def write_matrices_to_file(G, H,U, V, W, filepath):

    GG = pd.DataFrame(G)

    HH = pd.DataFrame(H)
    UU = pd.DataFrame(U)

    VV = pd.DataFrame(V)

    WW = pd.DataFrame(W)
    embeding_vector = (G + np.dot(U, W.T)) / 2


    EE=pd.DataFrame(embeding_vector)
    # 将 DataFrame 保存为 CSV 文件
    GG.to_csv(os.path.join(filepath, 'G.csv'), index=False, header=False)
    print("G saved to G.csv")
    UU.to_csv(os.path.join(filepath, 'U.csv'), index=False, header=False)
    print("U saved to U.csv")
    HH.to_csv(os.path.join(filepath, 'H.csv'), index=False, header=False)
    print("H saved to H.csv")
    VV.to_csv(os.path.join(filepath, 'V.csv'), index=False, header=False)
    print("V saved to V.csv")
    WW.to_csv(os.path.join(filepath, 'W.csv'), index=False, header=False)
    print("W saved to W.csv")
    EE.to_csv(os.path.join(filepath, 'EE.csv'), index=False, header=False)
    print("embeding vector saved to EE.csv")
def write_O1_O2_O3_to_file(O1, O2, O3,O_weights, filepath):
    O1 = pd.DataFrame(O1)
    O2 = pd.DataFrame(O2)
    O3 = pd.DataFrame(O3)
    # 将 DataFrame 保存为 CSV 文件
    filepath=os.path.join(filepath, str(O_weights))
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    O1.to_csv(os.path.join(filepath, 'O1.csv'), index=False, header=False)
    print("O1 saved to O1.csv")
    O2.to_csv(os.path.join(filepath, 'O2.csv'), index=False, header=False)
    print("O2 saved to O2.csv")
    O3.to_csv(os.path.join(filepath, 'O3.csv'), index=False, header=False)
    print("O3 saved to O3.csv")

def write_high_loss_node_id_to_file(high_loss_node_id, filepath):
    high_loss_node_id = pd.DataFrame(high_loss_node_id)
    # 将 DataFrame 保存为 CSV 文件
    high_loss_node_id.to_csv(os.path.join(filepath, 'extend_high_loss_node_id.csv'), index=False, header=False)
    print("high_loss_node_id saved to high_loss_node_id.csv")
def read_csv_to_numpy(filepath):
    # 读取 CSV 文件
    df = pd.read_csv(filepath, header=None)
    # 将 DataFrame 转换为 NumPy 数组
    array = df.to_numpy()
    return array