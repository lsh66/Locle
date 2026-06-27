import sys
import os

from torch.onnx.symbolic_opset11 import insert

from run_zu.test import queried_nodes
from run_zu.test_semi_ONE_cpu import SemiONEPipeline
# from run_zu.utils.process_citeseer import num_nodes

# 动态将 ~/total_GAT 添加到 PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime

import networkx as nx
from semi_ONE_cpu import read_csv_to_numpy
from utils.load_dataset import load_data
from utils.metric import evaluate_metrics

from advanced_util import (
    process_selected_nodes,
    redistribute_duplicates_to_max_AS_per,
    missing_nodes_to_community, parse_args, evaluate_on_test_nodes, get_divrank_nodes
)


class CommunityExpansionEvaluator:
    def __init__(self, raw_budget,top_indices, dataset_name, mini_dataset, content_file_path, diff_tol,theta,min_k):
        """
        初始化并加载数据
        """
        self.top_indices = top_indices
        self.min_k = min_k
        self.final_communitys = None
        self.theta =theta
        self.raw_budget = raw_budget
        self.metrics = None
        self.test_metrics = None
        self.queried_nodes = None
        self.dataset_name = dataset_name
        self.mini_dataset = mini_dataset
        # 加载数据集
        self.undirected_A, _, self.true_labels, _ = load_data(dataset_name, mini_dataset=self.mini_dataset)
        self.graph = nx.Graph(self.undirected_A)
        self.predicted_labels = []
        # 加载内容向量与高损失节点
        self.content = read_csv_to_numpy(content_file_path)
        self.selected_nodes = None
        self.c_labels = []
        self.n = len(self.true_labels)
        self.diff_tol = diff_tol


    def print_communities(self, communitys):
        for idx, community in enumerate(communitys):
            print(f"社区 {idx}: {community}")
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def run(self, max_iter=60, tol=1e-6):
        """
        运行社区贪心拓展、去重分配、补充遗漏节点并评估
        """
        # self.min_k=0
        # self.top_indices=[]

        # one_budget=0
        # expand_budget=0

        un_budget = self.raw_budget - self.min_k
        one_budget = int(un_budget*self.theta)
        # one_budget =un_budget
        expan_budget=un_budget-one_budget
        # expan_budget = 0
        # pipeline = SemiONEPipeline(dataset=self.dataset_name, mini_dataset=result_time_08281704_LOSS_397356.76515277824, result_dir=result_time_08281704_LOSS_397356.76515277824, K_iter=4,
        #                            alpha=0.2, beta=0.05, max_iter=100, tol=1e-2,
        #                            sigma1=1.0, sigma2=1.0, sigma3=1.0, one_budget=one_budget, weights=(2, 1.1, 0.5))

        # pipeline = SemiONEPipeline(
        #     dataset=self.dataset_name, mini_dataset=None, result_dir=None,
        #     K_iter=34,  # 建议比类别数多一些（6×17=102），保证分解空间足够
        #     alpha=0.7,  # 属性损失权重，低一点（避免高维特征压制结构）
        #     beta=0.01,  # 不一致性约束权重，高一点（帮助平衡结构/属性）
        #     max_iter=300,  # Wiki 数据更复杂，迭代次数适当增加
        #     tol=1e-3,  # 精度要求稍放宽
        #     sigma1=1.0, sigma2=1.0, sigma3=1.0,  # 高斯核缩放，sigma2大一些增强属性容忍度
        #     one_budget=one_budget,  # 查询预算可稍加大，社区更多
        #     weights=(2, 1.0,0.5)  # 三部分损失权重，不要让属性权重过大
        # )
        pipeline = SemiONEPipeline(
            dataset=self.dataset_name,
            mini_dataset=None,
            result_dir=None,
            # ---------------- 社区/矩阵分解维度 ----------------
            K_iter=2,  # BlogCatalog 类别少（6 类），K_iter=2 使得 NMF 或社区矩阵维度适中，避免 loss 爆炸
            # BlogCatalog 特征维度高，太大K会导致 loss 不稳定

            # ---------------- 损失权重 ----------------
            alpha=0.3,  # 结构损失略高，保持社区结构主导
            beta=0.01,  # 属性损失较小，特征高维且稀疏
            sigma1=1.0,  # 结构一致性权重
            sigma2=1.0,  # 属性一致性权重
            sigma3=0.5,  # 不一致性损失较小，避免爆炸

            # ---------------- 迭代与收敛 ----------------
            max_iter=100,  # 更多迭代，确保收敛
            tol=1e-3,  # 收敛阈值稍严格

            # ---------------- ONE 预算 ----------------
            one_budget=one_budget,  # 节点多，预算可稍大
            weights=(2.5, 1.0, 0.5)  # 主任务（结构）最重要
        )

        pipeline.load_dataset()


        if self.dataset_name=="cora":
            pipeline.result_dir= "../cora/result_time_07092043_LOSS_36656.50"
        if self.dataset_name=="citeseer":
            pipeline.result_dir= "../citeseer/result_time_07010920_LOSS_85351.35"
        if self.dataset_name=="pubmed":
            pipeline.result_dir= "../pubmed/result_time_08260112_LOSS_84908.93"
        if self.dataset_name=="photo":
            pipeline.result_dir= "../photo/result_time_08251821_LOSS_306045.83"
        if self.dataset_name=="wiki":
            pipeline.result_dir= "../wiki/result_time_08280253_LOSS_7427166.78"
        if self.dataset_name=="blogcatalog":
            pipeline.result_dir= "../blogcatalog/result_time_08282019_LOSS_397356.77"


        pipeline.load_embedding_from_dir(pipeline.result_dir) #加载GHUVW
        # pipeline.read_compute_O()
        pipeline.compute_outliers()
        #
        #
        high_loss_node_path=pipeline.extract_high_loss_nodes_and_save()
        high_loss_node_path=os.path.join(pipeline.result_dir,str(pipeline.weights),str(pipeline.one_budget))

        self.selected_nodes = read_csv_to_numpy(os.path.join(high_loss_node_path,"extend_high_loss_node_id.csv"))



        communitys, self.c_labels, expan_queried_count, self.queried_nodes, queried_nodes_to_community = process_selected_nodes(
            expan_budget,
            self.graph,
            self.dataset_name,

            self.mini_dataset,
            self.top_indices,
            self.selected_nodes,
            self.graph,
            self.content,
            max_iter=max_iter,
            tol=tol,
            diff_tol=self.diff_tol
        )
        print("self.c_labels: ", self.c_labels)
        print("queried_nodes_to_community",queried_nodes_to_community)
        print("queried_nodes: ", self.queried_nodes)
        print("扩展时用的预算",expan_queried_count)

        print("✅ 去重分配重复元素至 AS 最大社区...")
        no_dul_communitys, dup_num = redistribute_duplicates_to_max_AS_per(
            self.c_labels,
            queried_nodes_to_community,
            self.queried_nodes,
            communitys,
            self.content,
            self.graph
        )

        self.print_communities(no_dul_communitys)
        print("重复元素的个数：", dup_num)
        # 查找遗漏节点
        covered_nodes_set = set()
        covered_nodes_set.update(self.queried_nodes)
        for community in no_dul_communitys:
            covered_nodes_set.update(community)

        all_nodes_set = set(range(self.n))
        missing_nodes = sorted(list(all_nodes_set - covered_nodes_set))
        print(len(missing_nodes))  #2007
        print(len(covered_nodes_set)) #1305
        print(len(all_nodes_set))




        print(f" 发现 {len(missing_nodes)} 个遗漏节点，分配至最近社区...")

        # 将遗漏节点分配回社区
        self.final_communitys = missing_nodes_to_community(
            no_dul_communitys,
            covered_nodes_set,
            missing_nodes,
            self.graph,
            self.content
        )
        # self.print_communities(self.final_communitys)
    # 评估社区指标

        print("计算社区评估指标...")

        id_to_label = {}
        for index, community in enumerate(self.final_communitys):

            for nid in community:
                id_to_label[nid] = self.c_labels[index]

        sorted_dict = dict(sorted(id_to_label.items()))

        for key, value in sorted_dict.items():
            self.predicted_labels.append(value)

    # print("查询的节点总数：", queried_nodes_len)

        self.metrics = evaluate_metrics(self.predicted_labels, self.true_labels, graph=self.graph)

    # self.test_metrics = evaluate_on_test_nodes(self.graph,
    #                                            self.queried_nodes,
    #                                            self.predicted_labels,
    #                                            self.true_labels,
    #                                            self.c_labels,
    #                                            self.final_communitys,
    #                                            metric_record=result_time_08281704_LOSS_397356.76515277824,
    #                                            query_record=result_time_08281704_LOSS_397356.76515277824,
    #                                            )
    #     one budget+expan_queried_cout+min_k=总预算
        print("one_budget", one_budget)
        print("设置的expan_budget", expan_budget)
        print("使用的扩展预算", expan_queried_count)
        print("divrank标注的节点",self.min_k)
        print("使用的总预算：", self.min_k + one_budget + expan_queried_count)
        print("总预算占比：", (self.min_k + one_budget + expan_queried_count) / self.graph.number_of_nodes())

        for k, v in self.metrics.items():
            print(f"{k}: {v:.4f}")


    def get_communities(self):
        """
        返回最终社区结果
        """
        return self.final_communitys

    def get_metrics(self):
        """
        返回评估指标
        """
        return self.metrics


if __name__ == "__main__":
    # args = parse_args()
    dataset_name = "cora"
    # dataset_name = "citeseer"
    # dataset_name = "blogcatalog"
    # dataset_name = "photo"
    # raw_budgets=[164,330,663,996,1329,1663,1993,2326,2659,2992] citerseer
    num_nodes=2708
    # per_budgets=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  #0.05预算太小
    raw_budgets=[]
    per_budgets = [0.05]
    top_indices, min_k = get_divrank_nodes(dataset=dataset_name)
    #blogcatalog
    # top_indices=[0, 2, 1, 11, 7, 3, 9, 13, 4, 8, 87]
    # min_k=11


    print(top_indices)
    print(min_k)

    for i,r in enumerate(per_budgets):
        raw_budgets.append(int(r*num_nodes))
    print("raw_budgets: ", raw_budgets)
    # for i,r in enumerate(per_budgets):

    r=per_budgets[0]
    print("per_budget:",r)
    print("*******dataset:******", dataset_name)

    raw_budget = raw_budgets[0]
    print("raw_budget:",raw_budget)
    if raw_budget < min_k:
        print("出错，预算太小，无法做到label的全覆盖，预算为：", raw_budget, "最小div_k为：", min_k)

    else:
        theta=0.2
        print("theta",theta)


        mini_dataset = None
        content_file_path=f"data/{dataset_name}/content_file.csv"

        start_t = datetime.datetime.now()
        print("开始时间", start_t.strftime("%Y-%m-%d %H:%M:%S"))
        evaluator = CommunityExpansionEvaluator(
            raw_budget=raw_budget,
            top_indices=top_indices,
            dataset_name=dataset_name,
            mini_dataset=mini_dataset,

            content_file_path=content_file_path,
            diff_tol=0.3,
            theta=theta,
            min_k=min_k
        )
        #一般数据集取值60，但pubmed大 120
        evaluator.run(max_iter=60, tol=1e-3)

        communities = evaluator.get_communities()
        print("theta", theta)
        metrics = evaluator.get_metrics()

        end_t = datetime.datetime.now()
        print("结束时间", end_t.strftime("%Y-%m-%d %H:%M:%S"))
        run_t = end_t - start_t
        print("运行时间", run_t)
