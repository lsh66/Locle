import os

import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def compute_normalized_laplacian(adj_matrix):
    L = csgraph.laplacian(adj_matrix, normed=True)
    return L


def compute_structure_loss_laplacian(B, laplacian_matrix):
    LB = laplacian_matrix @ B
    L_s = np.trace(B.T @ LB)
    return L_s


def linear_representation_embedding(
    adj_matrix, attr_matrix, K, alpha=0.1, beta=0.1, gamma=0.01, lr=0.01, max_iter=200
):
    """
    改进版 linear_representation_embedding:
      - 使用归一化拉普拉斯矩阵结构损失
      - 自动梯度更新 B, U, V, W
      - 可视化损失曲线
    """
    N, D = attr_matrix.shape
    L_sym = compute_normalized_laplacian(adj_matrix)

    B = np.random.rand(N, N)
    np.fill_diagonal(B, 0)

    U = np.random.rand(N, K) * 0.01
    V = np.random.rand(K, D) * 0.01

    pca = PCA(n_components=K)
    loss_history = []

    for epoch in range(max_iter):
        # Forward
        L_s = compute_structure_loss_laplacian(B, L_sym)
        L_a = np.linalg.norm(attr_matrix - U @ V, ord='fro') ** 2
        G = adj_matrix @ B
        G = np.nan_to_num(G, nan=0.0, posinf=1e6, neginf=-1e6)
        G_max = np.max(np.abs(G))
        if G_max > 1e3:
            G = G / G_max

        G_reduced = pca.fit_transform(G)
        L_i = np.linalg.norm(G_reduced - U, ord='fro') ** 2

        reg_loss = np.sum(np.linalg.norm(B, axis=1))
        total_loss = L_s + alpha * L_a + beta * L_i + gamma * reg_loss

        loss_history.append(total_loss)

        # Backward (Simple gradient descent)
        grad_B = 2 * (L_sym @ B) + gamma * (B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8))
        # grad_U = -2 * alpha * (attr_matrix - U @ V) @ V.T - 2 * beta * (G_reduced - U)
        # grad_V = -2 * alpha * U.T @ (attr_matrix - U @ V)

        grad_U = -2 * alpha * (attr_matrix - U @ V) @ V.T - 2 * beta * (G_reduced - U)
        grad_V = -2 * alpha * U.T @ (attr_matrix - U @ V)

        # Gradient Clipping
        max_grad_norm = 1e3
        grad_U = np.clip(grad_U, -max_grad_norm, max_grad_norm)
        grad_V = np.clip(grad_V, -max_grad_norm, max_grad_norm)

        U -= lr * grad_U
        V -= lr * grad_V

        # Normalization
        U_norm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        U = U / U_norm

        V_norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        V = V / V_norm

        B -= lr * grad_B
        np.fill_diagonal(B, 0)
        # U -= lr * grad_U
        # V -= lr * grad_V

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    # Compute W via Procrustes
    G = adj_matrix @ B
    G = np.nan_to_num(G, nan=0.0, posinf=1e6, neginf=-1e6)
    G_max = np.max(np.abs(G))
    if G_max > 1e3:
        G = G / G_max
    G_reduced = pca.fit_transform(G)
    G_centered = G_reduced - np.mean(G_reduced, axis=0)
    U_centered = U - np.mean(U, axis=0)

    # X, _, Yt = np.linalg.svd(G_centered.T @ U_centered)

    svd_input = G_centered.T @ U_centered
    svd_input = np.nan_to_num(svd_input, nan=0.0, posinf=1e6, neginf=-1e6)
    svd_input += 1e-6 * np.eye(svd_input.shape[0])

    try:
        X, _, Yt = np.linalg.svd(svd_input)
    except np.linalg.LinAlgError:
        print("Warning: SVD did not converge, using np.eye fallback")
        X = np.eye(svd_input.shape[0])
        Yt = np.eye(svd_input.shape[0])

    W = X @ Yt

    # final_embedding = (G_reduced + U @ W.T) / 2
    final_embedding =U
    # 可视化损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve during Training')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #
    LOSS=loss_history[-1]
    return final_embedding, B, U, V, W, LOSS,G_reduced


def write_Advanced_ONE_matrix_to_file(B, U, V, G_reduced, W, embedding, filepath):
    BB = pd.DataFrame(B)
    UU = pd.DataFrame(U)
    VV = pd.DataFrame(V)
    GG = pd.DataFrame(G_reduced)
    WW = pd.DataFrame(W)
    em = pd.DataFrame(embedding)

    BB.to_csv(os.path.join(filepath, 'advanced_BB.csv'), index=False, header=False)
    UU.to_csv(os.path.join(filepath, 'advanced_UU.csv'), index=False, header=False)
    VV.to_csv(os.path.join(filepath, 'advanced_VV.csv'), index=False, header=False)
    GG.to_csv(os.path.join(filepath, 'advanced_GG.csv'), index=False, header=False)
    WW.to_csv(os.path.join(filepath, 'advanced_WW.csv'), index=False, header=False)
    em.to_csv(os.path.join(filepath, 'ONE_advanced_embedding_vector.csv'), index=False, header=False)
    print("ONE advanced embedding vector and BBUUVVGGWW are saved")


def identify_critical_nodes(B, C, U, V, G_reduced, W, alpha=1.0, beta=1.0, gamma=1.0,percentage = 0.10):
    """
    综合考虑：
    - B行范数（结构代表性）
    - 属性重建误差（逐节点）
    - 结构-属性不一致误差（逐节点）
    返回排序的关键节点索引和对应分数。
    """
    s_B = np.linalg.norm(B, axis=1)
    attr_recon = U @ V
    s_attr = np.linalg.norm(C - attr_recon, axis=1)
    UW = U @ W.T
    s_incon = np.linalg.norm(G_reduced - UW, axis=1)

    scores = alpha * s_B + beta * s_attr + gamma * s_incon
    num_critical = int(len(scores) * percentage)
    critical_nodes = np.argsort(scores)[::-1][:num_critical]
    print(f"Top {percentage * 100:.0f}% 关键节点数量: {num_critical}")
    print("关键节点索引:", critical_nodes)
    return critical_nodes, scores



# if __name__ == "__main__":
    # N = 100
    # K = 10
    # D = 20
    #
    # adj_matrix = np.random.rand(N, N)
    # adj_matrix = (adj_matrix + adj_matrix.T) / 2
    # np.fill_diagonal(adj_matrix, 0)
    # adj_matrix[adj_matrix < 0.95] = 0
    #
    # attr_matrix = np.random.rand(N, D)



    # embedding, B, U, V, W, loss_history,G_reduced= linear_representation_embedding(
    #     adj_matrix, attr_matrix, K,
    #     alpha=0.1, beta=0.1, gamma=0.01, lr=0.03, max_iter=350
    # )
    # critical_nodes, scores=identify_critical_nodes(B, attr_matrix, U, V, G_reduced, W, alpha=1.0, beta=1.0, gamma=1.0,percentage = 0.10)
    # print("Final embedding shape:", embedding.shape)  #Final embedding shape: (100, 10)
    # print("critical nodes:", critical_nodes)
    # print("scores:", scores)
