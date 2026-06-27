from run.utils.process_MVC import query_oracle


def process_merge_community(dataset,G,new_split_clusters2,new_split_clusters2_key_list,CL_sets_list):
    '''
    合并社区 每个社区抽象成一个节点，代表节点是簇心节点
    :param G:
    :param new_split_clusters2:
    :param new_split_clusters2_key_list:
    :param CL_sets_list: 两两之间都有CL边相连的簇心组成的数组
    :return:
    '''
    #快速合并

    remain_cens=remove_b_from_a(new_split_clusters2_key_list,CL_sets_list)  #没有经历过分裂的簇心数组
    remove_cens=[]
    CL_sets_list_add = [s.copy() for s in CL_sets_list]

    for index_r,r in enumerate(remain_cens):

        index_list=0
        for index,s in enumerate(CL_sets_list):
            CL_assigned = False
            for c in s:
                if(query_oracle(r,c,dataset=dataset)==1):
                    CL_assigned=True
                    break
            if CL_assigned==False:
                CL_sets_list_add[index].add(r)
                remove_cens.append(r)
    key_delete=[]
    remain_cens = [r for r in remain_cens if r not in remove_cens]

    for r in remain_cens:
        for index,s in enumerate(CL_sets_list_add):
            for c in s:
                if(query_oracle(r,c,dataset=dataset)==1):
                    new_split_clusters2[c].update(new_split_clusters2[r])
                    key_delete.append(r)
                    break
    for key in set(key_delete):
        if key in new_split_clusters2:
            del new_split_clusters2[key]
    new_split_clusters2_key_list=list(new_split_clusters2.keys())
    return new_split_clusters2,new_split_clusters2_key_list


def remove_b_from_a(a,b):
    '''
    从数组 a 中移除数组 b 中的元素
    :param a: a是一个数组
    :param b: b是由集合组成的一个数组
    :return:
    '''

    # 初始化一个空集合用于存储要去除的元素
    elements_to_remove = set()

    # 遍历数组 b 中的每个集合
    for sub_set in b:
        # 将集合中的元素添加到要去除的元素集合中
        elements_to_remove.update(sub_set)

    # 过滤掉数组 a 中存在于要去除的元素集合中的元素
    result = [x for x in a if x not in elements_to_remove]
    return result