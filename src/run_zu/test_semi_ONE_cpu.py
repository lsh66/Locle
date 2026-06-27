import datetime
import os
import time

import networkx as nx
import numpy as np
import torch

from semi_ONE_cpu import (
    one_embedding,
    compute_outlier_scores,
    compute_composite_outlier_score,
    extract_high_loss_nodes,
    write_matrices_to_file,
    write_high_loss_node_id_to_file,
    read_csv_to_numpy
)
from utils.load_dataset import load_data


class SemiONEPipeline:
    def __init__(self, dataset,mini_dataset, result_dir=None, K_iter=6,
                 alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
                 sigma1=1.0, sigma2=1.0, sigma3=1.0,one_budget=None,weights=(1,1,1)):
        self.budget = None
        self.dataset = dataset
        self.K_iter = K_iter
        self.alpha = alpha
        self.mini_dataset = mini_dataset
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.one_budget = one_budget
        self.result_dir = result_dir  # 可为空，自动生成
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Placeholders
        self.undirected_A = None
        self.C = None
        self.true_labels = None
        self.NoComm = None

        self.G = self.H = self.U = self.V = self.W = None
        self.O1 = self.O2 = self.O3 = None
        self.high_loss_node_ids = None
        self.weights = weights
        self.graph = None
    def load_dataset(self):
        self.undirected_A, _, self.true_labels, self.NoComm = load_data(self.dataset,mini_dataset=self.mini_dataset)
        self.C = np.array(_)  # 特征矩阵
        self.graph = nx.from_numpy_array(self.undirected_A)
    def run_embedding(self):
        print("[INFO] Running semi-supervised embedding...")
        self.G, self.H, self.U, self.V, self.W, loss = one_embedding(
            self.undirected_A, self.C, self.K_iter, self.NoComm,
            alpha=self.alpha, beta=self.beta,
            max_iter=self.max_iter, tol=self.tol
        )

        if self.result_dir is None:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%m%d%H%M")
            if self.mini_dataset is not None:
                self.result_dir = f"../{self.dataset}/{self.mini_dataset}/result_time_{timestamp}_LOSS_{loss:.2f}/"
            else:
                self.result_dir = f"{self.dataset}/result_time_{timestamp}_LOSS_{loss:.2f}/"

        os.makedirs(self.result_dir, exist_ok=True)
        write_matrices_to_file(self.G, self.H, self.U, self.V, self.W, self.result_dir)

    def load_embedding_from_dir(self, path: str):
        print(f"[INFO] Loading embedding matrices from {path}...")
        self.G = read_csv_to_numpy(os.path.join(path, "G.csv"))
        self.H = read_csv_to_numpy(os.path.join(path, "H.csv"))
        self.U = read_csv_to_numpy(os.path.join(path, "U.csv"))
        self.V = read_csv_to_numpy(os.path.join(path, "V.csv"))
        self.W = read_csv_to_numpy(os.path.join(path, "W.csv"))
        self.result_dir = path

    def compute_outliers(self):
        print("[INFO] Computing outlier scores...")
        self.O1, self.O2, self.O3 = compute_outlier_scores(
            self.undirected_A, self.C,
            self.G, self.H, self.U, self.V, self.W,
            sigma1=self.sigma1, sigma2=self.sigma2, sigma3=self.sigma3
        )
    def read_compute_O(self):
        path=os.path.join(self.result_dir,str(self.weights))
        self.O1 = read_csv_to_numpy(os.path.join(path, "O1.csv"))
        self.O2 = read_csv_to_numpy(os.path.join(path, "O2.csv"))
        self.O3 = read_csv_to_numpy(os.path.join(path, "O3.csv"))
         #compute_composite_outlier_score的weight
        compute_composite_outlier_score(self.O1, self.O2, self.O3,self.weights)
        # write_O1_O2_O3_to_file(self.O1, self.O2, self.O3, self.weights, self.result_dir)

    def extract_high_loss_nodes_and_save(self):
        print("[INFO] Extracting high-loss nodes...")

        # O1 = read_csv_to_numpy(os.path.join(r"../citeseer/result_time_07010920_LOSS_85351.35/(1.2, 1.5, 1)", "O1.csv"))
        # O2 = read_csv_to_numpy(os.path.join(r"../citeseer/result_time_07010920_LOSS_85351.35/(1.2, 1.5, 1)", "O2.csv"))
        # O3 = read_csv_to_numpy(os.path.join(r"../citeseer/result_time_07010920_LOSS_85351.35/(1.2, 1.5, 1)", "O3.csv"))

        self.high_loss_node_ids = extract_high_loss_nodes(

             self.O1, self.O2, self.O3,
            one_budget=self.one_budget,weights=self.weights,
            method='composite'
        )
        path=os.path.join(self.result_dir,str(self.weights),str(self.one_budget))
        if not os.path.exists(path):
            os.makedirs(path)
        write_high_loss_node_id_to_file(self.high_loss_node_ids, path)
        return path

    def run_all(self):
        self.load_dataset()
        self.run_embedding()
        # self.compute_outliers()
        # self.extract_high_loss_nodes_and_save()

# 方式 1：从头运行训练 + 保存
if __name__ == "__main__":
    start_t = datetime.datetime.now()
    print("开始时间", start_t.strftime("%Y-%m-%d %H:%M:%S"))

    pipeline = SemiONEPipeline(dataset='cora', mini_dataset=None, result_dir=None, K_iter=6,
                               alpha=1.0, beta=1.0, max_iter=40, tol=1e-2,
                               sigma1=1.0, sigma2=1.0, sigma3=1.0, one_budget=10, weights=(1.2, 1.1, 1))
    # pipeline = SemiONEPipeline(dataset='pubmed', mini_dataset=result_time_08281704_LOSS_397356.76515277824, result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=4,
    #                            alpha=0.2, beta=0.05, max_iter=100, tol= 1e-2,
    #                            sigma1=1.0, sigma2=1.0, sigma3=1.0, one_budget=10, weights=(2, 1.1, 0.5))
    # pipeline.run_all()

    # pipeline = SemiONEPipeline(
    #     dataset='wiki', mini_dataset=result_time_08281704_LOSS_397356.76515277824, result_dir=result_time_08281704_LOSS_397356.76515277824,
    #     K_iter=6,  # 建议比类别数多一些（6×17=102），保证分解空间足够
    #     alpha=0.05,  # 属性损失权重，低一点（避免高维特征压制结构）
    #     beta=0.1,  # 不一致性约束权重，高一点（帮助平衡结构/属性）
    #     max_iter=200,  # Wiki 数据更复杂，迭代次数适当增加
    #     tol=1e-3,  # 精度要求稍放宽
    #     sigma1=1.0, sigma2=2.0, sigma3=1.0,  # 高斯核缩放，sigma2大一些增强属性容忍度
    #     one_budget=20,  # 查询预算可稍加大，社区更多
    #     weights=(1.3, 1.0, 1.0)  # 三部分损失权重，不要让属性权重过大
    # )
    # pipeline = SemiONEPipeline(
    #     dataset="blogcatalog",
    #     mini_dataset=None,
    #     result_dir=None,
    #     # ---------------- 社区/矩阵分解维度 ----------------
    #     K_iter=2,  # BlogCatalog 类别少（6 类），K_iter=2 使得 NMF 或社区矩阵维度适中，避免 loss 爆炸
    #     # BlogCatalog 特征维度高，太大K会导致 loss 不稳定
    #
    #     # ---------------- 损失权重 ----------------
    #     alpha=0.3,  # 结构损失略高，保持社区结构主导
    #     beta=0.01,  # 属性损失较小，特征高维且稀疏
    #     sigma1=1.0,  # 结构一致性权重
    #     sigma2=1.0,  # 属性一致性权重
    #     sigma3=0.5,  # 不一致性损失较小，避免爆炸
    #
    #     # ---------------- 迭代与收敛 ----------------
    #     max_iter=100,  # 更多迭代，确保收敛
    #     tol=1e-3,  # 收敛阈值稍严格
    #
    #     # ---------------- ONE 预算 ----------------
    #     one_budget=30,  # 节点多，预算可稍大
    #     weights=(2.5, 1.0, 0.5)  # 主任务（结构）最重要
    # )
    #
    pipeline.run_all()
    end_t = datetime.datetime.now()

    print("结束时间", end_t.strftime("%Y-%m-%d %H:%M:%S"))
    run_t = end_t - start_t
    print("运行时间", run_t)
    # pipeline.compute_outliers()
    # pipeline.extract_high_loss_nodes_and_save()

# # # 方式 2：加载已有嵌入结果
# if __name__ == "__main__":
#     # pipeline = SemiONEPipeline(dataset='citeseer',mini_dataset=result_time_08281704_LOSS_397356.76515277824,result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=6,     #   composite_score1 = m*normcen+(1-m)*score m=越大，对中心性影响越大，对信息丰富影响越小
#     #              alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
#     #              sigma1=1.0, sigma2=1.0, sigma3=1.0,m=0.5, outlier_threshold=0.05,weights=(1.2, 1.5, 1))
#     # pipeline = SemiONEPipeline(dataset='pubmed', mini_dataset=result_time_08281704_LOSS_397356.76515277824, result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=4,
#     #                            alpha=0.2, beta=0.05, max_iter=100, tol=1e-2,
#     #                            sigma1=1.0, sigma2=1.0, sigma3=1.0, one_budget=10, weights=(2, 1.1, 0.5))
#     pipeline = SemiONEPipeline(
#         dataset="blogcatalog",
#         mini_dataset=result_time_08281704_LOSS_397356.76515277824,
#         result_dir="result_time_08281704_LOSS_397356.76515277824",
#         # ---------------- 社区/矩阵分解维度 ----------------
#         K_iter=2,  # BlogCatalog 类别少（6 类），K_iter=2 使得 NMF 或社区矩阵维度适中，避免 loss 爆炸
#         # BlogCatalog 特征维度高，太大K会导致 loss 不稳定
#
#         # ---------------- 损失权重 ----------------
#         alpha=0.3,  # 结构损失略高，保持社区结构主导
#         beta=0.01,  # 属性损失较小，特征高维且稀疏
#         sigma1=1.0,  # 结构一致性权重
#         sigma2=1.0,  # 属性一致性权重
#         sigma3=0.5,  # 不一致性损失较小，避免爆炸
#
#         # ---------------- 迭代与收敛 ----------------
#         max_iter=100,  # 更多迭代，确保收敛
#         tol=1e-3,  # 收敛阈值稍严格
#
#         # ---------------- ONE 预算 ----------------
#         one_budget=30,  # 节点多，预算可稍大
#         weights=(2.5, 1.0, 0.5)  # 主任务（结构）最重要
#     )
#
#     pipeline.load_dataset()
#     pipeline.result_dir= "../pubmed/result_time_08260112_LOSS_84908.93"
#     pipeline.load_embedding_from_dir("../pubmed/result_time_08260112_LOSS_84908.93")
#
#     pipeline.compute_outliers()
#     pipeline.read_compute_O()
#     pipeline.extract_high_loss_nodes_and_save()
