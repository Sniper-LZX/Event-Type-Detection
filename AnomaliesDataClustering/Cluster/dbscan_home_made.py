import numpy as np
from Utils import distance_metric as dm

UNCLASSIFIED = 0
NOISE = -1


# 计算数据点两两之间的距离
def getDistanceMatrix(datas):
    N,D = np.shape(datas)
    dists = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = dm.calsim_eud(vi, vj)
            # print(dists[i, j])
    return dists


#  寻找以点cluster_id 为中心，eps 为半径的圆内的所有点的id
def find_points_in_eps(point_id,eps,dists):
    index = (dists[point_id]<=eps)
    return np.where(index==True)[0].tolist()


# 聚类扩展
# dists ： 所有数据两两之间的距离  N x N
# labs :   所有数据的标签 labs N，
# cluster_id ： 一个簇的标号
# eps ： 密度评估半径
# seeds： 用来进行簇扩展的点
# min_points： 半径内最少的点数
def expand_cluster(dists, labs, cluster_id, seeds, eps, min_points):
    
    
    i = 0
    while i< len(seeds):
        # 获取一个临近点
        Pn = seeds[i]
        # 如果该点被标记为NOISE 则重新标记
        if labs[Pn] == NOISE:
            labs[Pn] = cluster_id
        # 如果该点没有被标记过
        elif labs[Pn] == UNCLASSIFIED:
            # 进行标记，并计算它的临近点 new_seeds
            labs[Pn] = cluster_id
            new_seeds = find_points_in_eps(Pn,eps,dists)
            
            # 如果 new_seeds 足够长则把它加入到seed 队列中
            if len(new_seeds) >=min_points:
                seeds = seeds + new_seeds
        
        i = i+1
        
        
def dbscan(datas, eps, min_points):
    
    # 计算 所有点之间的距离
    dists = getDistanceMatrix(datas)
    
    
    # 将所有点的标签初始化为UNCLASSIFIED
    n_points = datas.shape[0]
    labs = [UNCLASSIFIED]*n_points
    
    cluster_id = 0
    # 遍历所有点
    for point_id in range(0, n_points):
        # 如果当前点已经处理过了
        if not(labs[point_id] == UNCLASSIFIED):
            continue
            
        # 没有处理过则计算临近点
        seeds = find_points_in_eps(point_id,eps,dists)
        
        # 如果临近点数量过少则标记为 NOISE
        if len(seeds)<min_points:
            labs[point_id] = NOISE
        else:
            # 否则就开启一轮簇的扩张
            cluster_id = cluster_id+1
            # 标记当前点
            labs[point_id] = cluster_id
            expand_cluster(dists, labs, cluster_id, seeds,eps, min_points)
    return labs, cluster_id
    
   
  

   
    
    
    
    
    
    
    
    
    
    
    
    
    

    


        
        
        
       
    


    
