import os
import time
import datetime

import networkx as nx
import numpy as np
import torch

from semi_ONE_gpu import (
    one_embedding,
    compute_outlier_scores,
    compute_composite_outlier_score,
    extract_high_loss_nodes,
    write_matrices_to_file,
    write_O1_O2_O3_to_file,
    write_high_loss_node_id_to_file,
    read_csv_to_numpy, run_one_embedding_safe
)
from utils.get_centrality import centralissimo
from utils.load_dataset import load_data


class SemiONEPipeline:
    def __init__(self, dataset,mini_dataset, result_dir=None, K_iter=6,
                 alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
                 sigma1=1.0, sigma2=1.0, sigma3=1.0, one_budget=10,weights=(1,1,1)):
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

        # A = torch.tensor(self.undirected_A, dtype=torch.float32, device=self.device)
        # C = torch.tensor(self.C, dtype=torch.float32, device=self.device)
        # self.G, self.H, self.U, self.V, self.W, loss = one_embedding(
        #     A, C, self.K_iter, self.NoComm,
        #     alpha=self.alpha, beta=self.beta,
        #     max_iter=self.max_iter, tol=self.tol,device=self.device
        # )

        self.G, self.H, self.U, self.V, self.W, loss = run_one_embedding_safe(
            self.undirected_A, self.C,
            self.K_iter, self.NoComm,
            alpha=self.alpha, beta=self.beta,
            max_iter=self.max_iter, tol=self.tol,
            device=self.device
        )

        if self.result_dir is None:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%m%d%H%M")
            # if self.mini_dataset is not result_time_08281704_LOSS_397356.76515277824:
            #     self.result_dir = f"../{self.dataset}/{self.mini_dataset}/result_time_{timestamp}_LOSS_{loss:.2f}/"
            # else:
            self.result_dir = f"{self.dataset}/result_time_{timestamp}_LOSS_{loss:.2f}/"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        write_matrices_to_file(
            self.G.cpu(), self.H.cpu(),
            self.U.cpu(), self.V.cpu(),
            self.W.cpu(), self.result_dir
        )

    def load_embedding_from_dir(self, path: str):
        print(f"[INFO] Loading embedding matrices from {path} (CUDA)...")
        self.G = torch.tensor(read_csv_to_numpy(os.path.join(path, "G.csv")), device=self.device)
        self.H = torch.tensor(read_csv_to_numpy(os.path.join(path, "H.csv")), device=self.device)
        self.U = torch.tensor(read_csv_to_numpy(os.path.join(path, "U.csv")), device=self.device)
        self.V = torch.tensor(read_csv_to_numpy(os.path.join(path, "V.csv")), device=self.device)
        self.W = torch.tensor(read_csv_to_numpy(os.path.join(path, "W.csv")), device=self.device)
        self.result_dir = path

    def compute_outliers(self):
        print("[INFO] Computing outlier scores (CUDA)...")
        self.O1, self.O2, self.O3 = compute_outlier_scores(
            torch.tensor(self.undirected_A, dtype=torch.float32, device=self.device),
            torch.tensor(self.C, dtype=torch.float32, device=self.device),
            self.G, self.H, self.U, self.V, self.W,
            sigma1=self.sigma1, sigma2=self.sigma2, sigma3=self.sigma3,
            device=self.device
        )
        write_O1_O2_O3_to_file(
            self.O1.cpu(), self.O2.cpu(), self.O3.cpu(),
            self.weights, self.result_dir
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
            # self.graph,O1, O2, O3,
             self.O1, self.O2, self.O3,
            one_budget=self.one_budget,weights=self.weights,
            method='composite',device=self.device
        )
        path=os.path.join(self.result_dir,str(self.weights),str(self.one_budget))
        if not os.path.exists(path):
            os.makedirs(path)
        write_high_loss_node_id_to_file(self.high_loss_node_ids.cpu(), path)


    def run_all(self):
        self.load_dataset()
        self.run_embedding()
        # self.compute_outliers()
        # self.extract_high_loss_nodes_and_save()


# 方式 1：从头运行训练 + 保存
if __name__ == "__main__":
    start_t = datetime.datetime.now()
    print("开始时间", start_t.strftime("%Y-%m-%d %H:%M:%S"))
    # pipeline = SemiONEPipeline(dataset='webkbs', mini_dataset="texas", result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=6,
    #                            alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
    #                            sigma1=1.0, sigma2=1.0, sigma3=1.0, outlier_threshold=0.60, weights=(1.2,1.1,1))

    # pipeline = SemiONEPipeline(dataset='pubmed', mini_dataset=result_time_08281704_LOSS_397356.76515277824, result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=6,
    #                            alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
    #                            sigma1=1.0, sigma2=1.0, sigma3=1.0, outlier_threshold=0.20, weights=(1.2, 1.1, 1))

    # pipeline = SemiONEPipeline(dataset='cora', mini_dataset=result_time_08281704_LOSS_397356.76515277824, result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=6,
    #                            alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
    #                            sigma1=1.0, sigma2=1.0, sigma3=1.0, one_budget=10, weights=(1.2, 1.1, 1))
    # pipeline.run_all()

    pipeline = SemiONEPipeline(dataset='photo', mini_dataset=None, result_dir=None, K_iter=6,
                               alpha=0.2, beta=0.05, max_iter=100, tol=1e-2,
                               sigma1=1.0, sigma2=5.0, sigma3=1.0, one_budget=10, weights=(2, 1.1, 0.5))
    pipeline.run_all()

# 计算完O值后提取高损失节点
# # # 方式 2：加载已有嵌入结果
# if __name__ == "__main__":
#     pipeline = SemiONEPipeline(dataset='citeseer',mini_dataset=result_time_08281704_LOSS_397356.76515277824,result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=6,     #   composite_score1 = m*normcen+(1-m)*score m=越大，对中心性影响越大，对信息丰富影响越小
#                  alpha=1.0, beta=1.0, max_iter=60, tol=1e-2,
#                  sigma1=1.0, sigma2=1.0, sigma3=1.0,m=0.5, outlier_threshold=0.05,weights=(1.2, 1.5, 1))
#     pipeline.load_dataset()
#     pipeline.result_dir= "../citeseer/result_time_07010920_LOSS_85351.35"
#     # pipeline.load_embedding_from_dir("../cora/result_time_07092043_LOSS_36656.50")
#     pipeline.read_compute_O()
#     # pipeline.compute_outliers()
#     pipeline.extract_high_loss_nodes_and_save()
    end_t = datetime.datetime.now()
    print("结束时间", end_t.strftime("%Y-%m-%d %H:%M:%S"))
    run_t = end_t - start_t
    print("运行时间", run_t)



