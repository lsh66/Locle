from collections import defaultdict

import numpy as np


from run.utils.load_dataset import constraint_transfer
from run.utils.process_MVC import query_oracle
def check_duplicate_assignments(cluster_dict):
    node_to_keys = defaultdict(set)

    for key, node_set in cluster_dict.items():
        for node in node_set:
            if node ==591:
                node_to_keys[node].add(key)
            if node ==641:
                node_to_keys[node].add(key)
            if node==642:
                node_to_keys[node].add(key)
    for node, keys in node_to_keys.items():
        if len(keys) > 1:
            print(f"Node {node} is assigned to multiple keys: {keys}")


def judge_node(G,  EE, alpha, beta,cluster,active_nodes,center_id):
    '''
    判断节点位置，是否是密集区还是边界区
    :param G:
    :param EE:
    :param alpha:
    :param beta:
    :param cluster: 当前处理的簇
    :param active_nodes: 当前处理的簇中的活跃节点
    :param center_id: 当前处理簇的簇心
    :return: 返回cluster簇中的活跃节点的密集近核节点和稀疏边界节点
    '''
    # for n in G.nodes():
    #     print(n)
    #     if n==988:
    #         print("find 988")
    #         print(G.neighbors(n))
    #         break



    near_core_dense_nodes=[]
    sparse_boundary_nodes=[]

    for node_id in active_nodes:
        print(node_id)
        Nu = list(G.neighbors(node_id))
        num_Nu= len(Nu)
        sum_jiao=0
        for n in cluster:
            if n in Nu:
                sum_jiao+=1
        if num_Nu==0: #活跃节点没有邻居节点,存储在sparse_boundary_nodes

            sparse_boundary_nodes.append(node_id)
            continue
        local_centrality= sum_jiao/num_Nu
        node_center_sim=cosine_similarity_vec(EE[node_id],EE[center_id])
        score=alpha*local_centrality+(1-alpha)*node_center_sim
        if score>beta:
            near_core_dense_nodes.append(node_id)
            # process_near_core_node_dense(G,node_id,cluster,center_id,EE) #中心密集区
        else:
            sparse_boundary_nodes.append(node_id)
    print("密集近核节点：",near_core_dense_nodes)
    print("稀疏边界节点：",sparse_boundary_nodes)
    return near_core_dense_nodes,sparse_boundary_nodes
def cosine_similarity_vec(a, b):
    a = np.array(a)
    b = np.array(b)
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0  # 避免除零错误
    else:
        return numerator / denominator
def process_near_core_node_dense(dataset,G,cluster,center_id,EE,near_core_dense_nodes):
    '''
    处理中心密集区节点
    :param G:
    :param cluster: 当前处理的簇
    :param center_id: 当前处理簇的簇心
    :param EE:
    :param near_core_dense_nodes: 当前簇活跃节点位于核心密集取的列表
    :return:
    '''
    # 创建一个 defaultdict，默认值为集合
    new_split_clusters = defaultdict(set)
    # 返回 CL_set 保存分裂簇时簇心之间是CL的簇心，两两之间都有CL边相连，用于指导快速社区合并

    CL_set=set()
    # 已经能保证在访问不存在的键时自动创建一个空集合

    # 第一阶段：根据 near_core_dense_nodes 构建初始的 new_split_clusters
    for node_id in near_core_dense_nodes:
        is_ML=query_oracle(node_id,center_id,dataset=dataset)
        if is_ML==1:
            new_split_clusters[center_id].add(node_id)
        else:
            new_split_clusters[node_id]=set()
    # 第二阶段，修改字典 与簇心是CL关系的节点间，如果是ML就合并，可以减少后期进行社区合并时的计算量
    keys_to_delete = []
    for key1 in list(new_split_clusters.keys()):
        for key2 in list(new_split_clusters.keys()):
            if key1 < key2 and center_id not in (key1, key2):
                is_ML = query_oracle(key1, key2,dataset=dataset)
                if is_ML == 1:
                    new_split_clusters[key1].add(key2)
                    keys_to_delete.append(key2)
    CL_set.update(new_split_clusters.keys())
    # 统一删除要删除的键
    for key in set(keys_to_delete):
        if key in new_split_clusters:
            del new_split_clusters[key]

    # 计算 new_split_clusters 中键的数量，即新的和之前簇心的数量 l+1
    key_count = len(new_split_clusters.keys())
    #第三阶段，给该簇中的节点划分从news_split_cluster.keys()划分归属
    out_nodes,new_split_clusters1=split_cluster(dataset,key_count,new_split_clusters,cluster)
    return out_nodes,new_split_clusters1,CL_set



    # is_ML=query_oracle(node,center_id)
    # if is_ML==1: #ML
    #
    #     # ML.append((node,center_id))
    #     # new_ML,new_CL=constraint_transfer(ML,CL=[])
    # else:

        # CL.append((node, center_id))
        # new_M+L,new_CL=constraint_transfer(ML=[],CL)
def split_cluster(dataset,key_count,new_split_clusters,cluster):
    '''
    划分簇
    :param key_count: 簇心的数量 l+1
    :param new_split_clusters: 字典，键值分别为簇心与节点的集合
    :param cluster: 当前簇
    :return:
    out_nodes: 离群点
    new_split_clusters: 新的簇心与节点的集合
    '''
    out_nodes=set()
    en_nodes=set()
    new_cens=list(new_split_clusters.keys())
    for key in new_cens:
        en_nodes.add(key)
        en_nodes.update(new_split_clusters[key])
    remain_nodes = set(cluster) - set(en_nodes)
    for i in remain_nodes:
        assigned = False
        for j in new_cens:
            if query_oracle(i,j,dataset=dataset) == 1:
                new_split_clusters[j].add(i)
                assigned = True
                break
        if not assigned:#与簇心都不是ML，那就标记为离群点
            out_nodes.add(i)

    return out_nodes,new_split_clusters

# def process_sparse_boundary_node(G,cluster,center_id,EE,sparse_boundary_nodes):
#     """处理边界稀疏区的活跃节点"""


def calculate_cluster_active_score(dataset,new_cens,cluster,clusters_active_nodes,G, EE, alpha=0.5, beta=0.7):
    '''
    计算簇的活跃分数
    :param new_cens: 簇心的索引
    :param cluster: 列表来存储每个簇的点的索引，簇的个数为行，每个簇的中点的索引为列
    :param new_cens: 所有簇心的最近节点的索引
    :param cluster:
    :param clusters_active_nodes: 每个簇中活跃节点的索引


    :param G:
    :param EE:
    :param alpha:
    :param beta:
    :return:
    '''
    out_nodes_all=set()
    new_split_clusters2 = defaultdict(set)
    CL_sets_list=[] #由集合组成的数组
    sparse_boundary_nodes_all=[]  #记录簇中无簇心紧密区活跃节点的位于稀疏边界区的活跃节点列表
    for i in range(len(cluster)):
        if len(cluster[i]) == 1:
            node_id = list(cluster[i])[0] if isinstance(cluster[i], set) else cluster[i][0]
            new_split_clusters2[new_cens[i]].add(node_id)
            sparse_boundary_nodes_all.append([])
            continue

        active_score=len(clusters_active_nodes[i])/len(cluster[i])
        print("第",i,"个的活跃节点占比:",active_score)
        # if active_score > 0.4:  #活跃节点在簇中占比过高，解体
        #     # sparse_boundary_nodes_all.append([])
        #     pass        #分解
        # else:
        # print("clusters_active_nodes[i]:",clusters_active_nodes[i])
        near_core_dense_nodes,sparse_boundary_nodes=judge_node(G,  EE, alpha, beta,cluster=cluster[i],active_nodes=clusters_active_nodes[i],center_id=new_cens[i]) #判断节点位置


        if len(near_core_dense_nodes)>0:  #簇中存在簇心紧密区的活跃节点
            sparse_boundary_nodes_all.append([])
            out_nodes,new_split_clusters1,CL_set=process_near_core_node_dense(dataset=dataset,G=G, cluster=cluster[i], center_id=new_cens[i], EE=EE, near_core_dense_nodes=near_core_dense_nodes)

            out_nodes_all.update(out_nodes)
            new_split_clusters2.update(new_split_clusters1)
            CL_sets_list.append(CL_set)
        else: #直接添加簇
            new_split_clusters2[new_cens[i]] = set(cluster[i])

            sparse_boundary_nodes_all.append(sparse_boundary_nodes)
    # 处理各个簇中的out——nodes（簇中与新簇心不是ML的情况）   包含跨社区的连接
    print("203行")
    # check_duplicate_assignments(new_split_clusters2)
    total_size = sum(len(v) for v in new_split_clusters2.values())
    print("total_size", total_size)
    new_split_clusters2_key_list = list(new_split_clusters2.keys())    #经过分裂以及处理过簇心紧密活跃节点后的，图中所有簇的簇心节点ID组成的数组
    for out in out_nodes_all:

        assigned = False
        for c in new_split_clusters2_key_list:
            is_ML=query_oracle(out, c,dataset=dataset)

            if is_ML==1:
                new_split_clusters2[c].add(out)
                assigned = True
                break
        if assigned==False:
            max_M_prime, community_core_id= find_community(G, out, new_split_clusters2_key_list, EE,
                                                                         new_split_clusters2, k=5)
            # print("out",type(out))  #out是一个整数
            # print("community_cen",community_cen)   #community_cen是一个集合
            # new_split_clusters2[community_core_id].add(out)
            remove_node_from_all_clusters(out, new_split_clusters2, keep=None)
            new_split_clusters2[community_core_id].add(out)

    # print("224行")
    # check_duplicate_assignments(new_split_clusters2)
    total_size = sum(len(v) for v in new_split_clusters2.values())
    print("total_size", total_size)
    #经过分裂以及处理过簇心紧密活跃节点后的，图中所有簇的簇心节点ID组成的数组
    for index,list1 in enumerate(sparse_boundary_nodes_all):
        if len(list1)>0:
            center_id=new_split_clusters2_key_list[index]
            for a in list1:
                is_ML=query_oracle(a,center_id,dataset=dataset)
                if is_ML==1:
                    pass #增强连接 暂不考虑
                else:
                    # pass#断开连接 暂不考虑
                    # 边界稀疏区的节点分配标签时，节点特征影响更大，此外，活跃节点位于稀疏边界区的比例大于簇心紧密区
                    #取top K的个簇心与查询节点相似度从高到低排序的簇，然后再看点与簇的连接数,以及ein/emou的增加还是减少，判断分配到哪个社区
                    max_M_prime, community_core_id=find_community(G,a,new_split_clusters2_key_list,EE,new_split_clusters2,k=5)

                    remove_node_from_all_clusters(a, new_split_clusters2, keep=None)
                    new_split_clusters2[community_core_id].add(a)
    print("244行")
    # check_duplicate_assignments(new_split_clusters2)
    total_size = sum(len(v) for v in new_split_clusters2.values())
    print("total_size", total_size)

    return new_split_clusters2,new_split_clusters2_key_list,CL_sets_list
def remove_node_from_all_clusters(a, clusters, keep=None):
    for key in clusters:
        if key != keep:
            clusters[key].discard(a)


def top_k_with_indices(arr, k):
    # 对数组元素及其索引进行排序
    sorted_data = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    # 取出前 k 个元素
    top_k = sorted_data[:k]
    # 分离出前 k 个元素的索引和值
    indices = [index for index, _ in top_k]
    values = [value for _, value in top_k]
    return indices, values


def calculate_delta_M(G,C,a):
    # 计算原始社区 C 的内边数和外边数

    ein = G.subgraph(C).number_of_edges()

    eout = 0
    for node in C:
        for neighbor in G.neighbors(node):
            if neighbor not in C:
                eout += 1

    # 计算新节点 a 与社区 C 内节点的连边数 kin 和与社区 C 外节点的连边数 kout
    kin = 0
    kout = 0
    for neighbor in G.neighbors(a):
        if neighbor in C:
            kin += 1
        else:
            kout += 1

    # 计算加入新节点后社区 C 的 M 值
    M_prime = (ein + kin) / (eout - kin + kout + 1e-8)


    return M_prime




def find_community(G,a,new_split_clusters2_key_list,EE,new_split_clusters2,k=5):
    '''
    :param k: 取top K的个簇心与查询节点相似度从高到低排序的簇
    :param a: 查询的位于稀疏边界区活跃节点
    :param new_cens2: 现有的所有簇心ID
    :param EE:
    new_split_clusters2:
    :return:

    '''
    max_M_prime = float('-inf')
    max_index = None
    sims=[]

    for index,center_id in enumerate(new_split_clusters2_key_list):
        sims.append(cosine_similarity_vec(EE[a],EE[center_id]))
    #取前k个大的数值及对应的索引
    indices, values=top_k_with_indices(sims,k)
    #计算加入后的deltaM
    for i in indices:
        M_prime=calculate_delta_M(G,new_split_clusters2[new_split_clusters2_key_list[i]],a)
        if M_prime > max_M_prime:
            max_M_prime = M_prime
            max_index = i

    community_core_id=new_split_clusters2_key_list[max_index]
    return max_M_prime,community_core_id



        #     pass        #分解
    #     # else:
    #     #     if sparse_boundary_nodes.length()==0:
    #     #         process_sparse_boundary_node(G, cluster=cluster[i], center_id=new_cens[i], EE=EE, sparse_boundary_nodes=sparse_boundary_nodes)