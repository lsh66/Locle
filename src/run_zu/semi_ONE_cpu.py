import os

import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.decomposition import NMF

from utils.get_centrality import centralissimo


def update_G(G,A, H, U, W,beta):
    """更新结构嵌入矩阵G"""

    for i in range(G.shape[0]):
        for k in range(G.shape[1]):
            Gik_numer = np.dot(H[k, :],(A[i, :] - (np.matmul(G[i], H) - np.multiply(
                                                                             G[i, k], H[k, :])))) + \
                        beta  * np.dot(U[i], W[k, :])
            Gik_denom = np.dot(H[k, :], H[k, :]) + beta

            G[i, k] = Gik_numer / Gik_denom


    return G


def update_U(U,C, V, G, W, beta):
    """更新属性嵌入矩阵U"""
    for i in range(U.shape[0]):
        #属性维度
        for k in range(U.shape[1]):
            Uik_numer_1 = beta *  (np.dot(V[k, :],(C[i] - (np.matmul(U[i, :], V) - np.multiply(
                                                                               U[i, k], V[k, :])))))
            # ONE代码里的
            Uik_numer_2 =   np.dot((G[i, :] - (np.matmul(U[i, :], W) - np.multiply(U[i, k], W[:, k]))), W[:, k])
            # 论文里的
            # Uik_numer_2 = gamma * (G[i, :] - (np.matmul(U[i, :], W) - np.multiply(U[i, k], W[:, k])))
            Uik_denom = beta *  np.dot(V[k, :], V[k, :]) +  np.dot(W[:, k], W[:, k])

            U[i, k] = (Uik_numer_1 + Uik_numer_2) / Uik_denom

    return U
# 辅助函数

def update_H(A, G,H):
    """更新结构嵌入辅助矩阵H"""
    for k in range(H.shape[0]):
        for j in range(H.shape[1]):

            # np.dot 当输入为一维数组时，它计算的是两个向量的内积（数量积），得到一个标量结果。
            # 当输入为二维数组时，它执行的是矩阵乘法，结果是一个新的二维矩阵。


            Hkj_numer = np.dot(G[:, k], (A[:, j] - (np.matmul(G, H[:, j]) - np.multiply(G[:, k], H[k, j]))))

            # Hkj_numer = alpha * np.dot(np.multiply(np.log(np.reciprocal(outl1)), G[:, k]), \
            #                            (A[:, j] - (np.matmul(G, H[:, j]) - np.multiply(G[:, k], H[k, j]))))
            # np.multiply 对元素进行一一对应相乘。
            Hkj_denom =  sum(np.multiply(G[:, k], G[:, k]))

            H[k, j] = Hkj_numer / Hkj_denom

    return H


def update_V(C, V,U):
    """更新属性嵌入辅助矩阵V"""
    for k in range(V.shape[0]):
        for d in range(V.shape[1]):
            Vkd_numer = np.dot(U[:, k], (C[:, d] - (np.matmul(U, V[:,d]) - np.multiply(U[:, k], V[k, d]))))
            Vkd_denom = sum(np.multiply(U[:, k], U[:, k]))

            V[k][d] = Vkd_numer / Vkd_denom

    return V


def update_W(U, G):
    """更新变换矩阵W"""
    # M = np.dot(U.T, G)
    # U_svd, _, Vt_svd = svd(M)
    # W = np.dot(U_svd, Vt_svd)


    svd_matrix = np.matmul(G.T, U)

    svd_u, svd_sigma, svd_vt = np.linalg.svd(svd_matrix)
    W = np.matmul(svd_u, svd_vt)
    return W



def calculate_loss(A, C, G, H, U, V, W, alpha, beta):
    """计算总损失函数值"""
    # 结构损失
    struct_loss = np.sum((A - np.dot(G, H)) ** 2)

    # 属性损失
    attr_loss = np.sum((C - np.dot(U, V)) ** 2)

    # 不一致损失
    dis_loss = np.sum((G - np.dot(U, W.T)) ** 2)

    # 总损失
    total_loss = struct_loss + alpha * attr_loss + beta * dis_loss

    return total_loss


def one_embedding(A, C, K_iter,NoComm,beta=1.0, alpha=1.0, max_iter=100, tol=1e-4):
    """
    实现异常感知网络嵌入(ONE)的迭代更新过程


    参数:
    A: 邻接矩阵 (n_nodes x n_nodes)
    C: 节点属性矩阵 (n_nodes x n_attrs)
    K: 嵌入维度
    alpha: 属性损失权重
    beta: 不一致损失权重
    max_iter: 最大迭代次数
    tol: 收敛容差

    返回:
    G: 结构嵌入矩阵
    H: 结构嵌入辅助矩阵
    U: 属性嵌入矩阵
    V: 属性嵌入辅助矩阵
    W: 变换矩阵
    """
    n_nodes = A.shape[0]
    n_attrs = C.shape[1]
    # NoComm代表数据集中不同类别的数量，也就是真实标签集合里不同元素的数量。K表示非负矩阵分解后矩阵的维度数量
    K = K_iter * NoComm
    # 初始化
    model = NMF(n_components=K, init='random', random_state=0, max_iter=1000)
    G = model.fit_transform(A)
    H = model.components_

    model = NMF(n_components=K, init='random', random_state=0, max_iter=1000)
    U = model.fit_transform(C)
    V = model.components_

    # # G = np.random.rand(n_nodes, K)
    # H = np.random.rand(K, n_nodes)
    # # U = np.random.rand(n_nodes, K)
    # V = np.random.rand(K, n_attrs)
    W = np.eye(K)

    prev_loss = float('inf')

    for iter in range(max_iter):
        # 迭代更新
        G = update_G(G, A, H, U, W, beta)
        H = update_H(A, G, H)
        U = update_U(U, C, V, G, W, beta)
        V = update_V(C, V, U)
        W = update_W(U, G)

        # 计算损失
        current_loss = calculate_loss(A, C, G, H, U, V, W, alpha, beta)

        # 检查收敛性
        loss_diff = prev_loss - current_loss
        if iter > 0 and abs(loss_diff) < tol:
            print(f"Converged at iteration {iter} with loss: {current_loss}")
            break
        if loss_diff<0:
            print("已经到极小值")
            break
        prev_loss = current_loss


        print(f"Iteration {iter}, Loss: {current_loss}")
        LOSS=current_loss
    return G, H, U, V, W,LOSS



# 事后计算异常得分
def compute_outlier_scores(A, C, G, H, U, V, W, sigma1=1.0, sigma2=1.0, sigma3=1.0):
    n_nodes = A.shape[0]
    O1 = np.zeros(n_nodes)  # 结构异常得分
    O2 = np.zeros(n_nodes)  # 属性异常得分
    O3 = np.zeros(n_nodes)  # 不一致异常得分

    for i in range(n_nodes):
        # 计算结构重构误差
        struct_error = np.sum((A[i] - G[i] @ H) ** 2)
        O1[i] = np.exp(-struct_error / (2 * sigma1 ** 2))

        # 计算属性重构误差
        attr_error = np.sum((C[i] - U[i] @ V) ** 2)
        O2[i] = np.exp(-attr_error / (2 * sigma2 ** 2))

        # 计算不一致误差
        dis_error = np.linalg.norm(G[i] - U[i] @ W) ** 2
        O3[i] = np.exp(-dis_error / (2 * sigma3 ** 2))

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
    threshold: 阈值（分位数时为百分比，如0.05表示后5%）
    method: 'composite'（综合得分）、'structural'（结构）、'attribute'（属性）、'discrepancy'（不一致）

    返回:
    high_loss_indices: 高损失节点索引列表
    """
    O1 = np.array(O1).flatten()
    O2 = np.array(O2).flatten()
    O3 = np.array(O3).flatten()

    if method == 'composite':
        # 计算综合得分
        composite_score = compute_composite_outlier_score(O1, O2, O3, weights)
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