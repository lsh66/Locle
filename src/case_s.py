import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch



# 这个文件是一个 t-SNE 可视化脚本，用于将高维特征矩阵降维到 2D 并绘图，同时展示聚类中心。
# 核心函数
# plot_tsne_with_centers：接收三个参数：
# feature_matrix：节点特征矩阵（numpy）
# label_tensor：节点标签（numpy）
# center_tensor：聚类中心（numpy）
# 流程：
# 将特征矩阵和聚类中心拼接后，一起做 t-SNE 降维到 2 维
# 分离出节点嵌入和中心嵌入
# 用散点图绘制节点（按标签着色，jet 色图）
# 用 x 标记绘制聚类中心点
# 显示并保存为 tsneeee.png


def plot_tsne_with_centers(feature_matrix, label_tensor, center_tensor):
    # Compute T-SNE embeddings for the feature matrix
    tsne = TSNE(n_components=2, random_state=42)
    aug_feature_matrix = np.concatenate((feature_matrix, center_tensor), axis=0)
    aug_embedded_features = tsne.fit_transform(aug_feature_matrix)
    embedded_features = aug_embedded_features[:feature_matrix.shape[0]]
    # center_tensor = aug_embedded_features[feature_matrix.shape[0]:]
    # import ipdb; ipdb.set_trace()
    # Plot nodes based on T-SNE embeddings and color by class labels
    plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=label_tensor, cmap=plt.cm.get_cmap("jet", np.unique(label_tensor).size), marker='o')
    
    # Compute T-SNE embeddings for the center tensor

    # embedded_centers = tsne.fit_transform(center_tensor)
    embedded_centers = aug_embedded_features[feature_matrix.shape[0]:]
    
    # Plot clustering centers with a different shape
    for idx, center in enumerate(embedded_centers):
        plt.scatter(center[0], center[1], color=plt.cm.jet(idx / np.unique(label_tensor).size), marker='x', s=100, edgecolors='k')
    
    # Display the plot
    plt.colorbar(ticks=range(np.unique(label_tensor).size))
    plt.clim(-0.5, np.unique(label_tensor).size - 0.5)
    plt.show()
    plt.savefig("tsneeee.png")




if __name__ == '__main__':
    data = torch.load("../..//ogb/preprocessed_data/new/cora_fixed_sbert.pt")
    feature_matrix = data.x.numpy()
    label_tensor = data.y.numpy()
    center = torch.load("../../ogb/preprocessed_data/aax/center_x_2708_7.pt")
    center = center.numpy()
    plot_tsne_with_centers(feature_matrix, label_tensor, center)