import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.transforms import BaseTransform
import scipy.sparse.linalg as sla
import scipy.linalg as la
try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
except Exception:
    Chem = None
    BRICS = None
from sklearn.cluster import KMeans, SpectralClustering
import scipy.sparse as sp

def build_adj_list(edge_index: torch.Tensor, num_nodes: int):
    """
    使用 Scipy 优化邻接表构建，比纯 Python 循环快 10-100 倍
    """
    # 转移到 CPU 并转为 numpy
    row, col = edge_index.cpu().numpy()
    
    # 构建稀疏矩阵 (C00 format -> LIL format 适合按行索引)
    # data 全为 1，形状为 [num_nodes, num_nodes]
    data = np.ones(len(row), dtype=np.int32)
    adj_mat = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    #以此格式转换列表
    adj_lil = adj_mat.tolil()
    
    # 转换为 dict list (这是 random walk 函数需要的格式)
    # rows 是一个 list of lists
    return adj_lil.rows 

def k_step_random_walk(adj_list, start_node: int, k: int) -> List[int]:
    walk_path = [start_node]
    current_node = start_node
    
    for step in range(k):
        # 【修改这里】
        # 1. 先判断是否为字典 (旧逻辑)
        if isinstance(adj_list, dict):
            neighbors = adj_list.get(current_node, [])
        # 2. 如果不是字典，则认为是 list 或 numpy.ndarray (新逻辑)
        #    Scipy 的 adj_lil.rows 返回的是 numpy array，直接用索引访问即可
        else:
            neighbors = adj_list[current_node]
            
        if not neighbors: # 空邻居处理
            walk_path.append(current_node)
        else:
            prev_node = walk_path[-2] if len(walk_path) > 1 else None
            if prev_node is not None and len(neighbors) > 1:
                # 尝试不往回走
                candidates = [n for n in neighbors if n != prev_node]
                if not candidates:
                    candidates = neighbors
            else:
                candidates = neighbors

            next_node = random.choice(candidates)
            walk_path.append(next_node)
            current_node = next_node
    
    return walk_path


def n_times_k_step_random_walks(edge_index: torch.Tensor, start_node: int, 
                                n: int, k: int, num_nodes: int, 
                                device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = edge_index.device

    adj_list = build_adj_list(edge_index, num_nodes)
    walks = []
    for _ in range(n):
        walk_path = k_step_random_walk(adj_list, start_node, k)
        walks.append(walk_path)
    
    return torch.tensor(walks, dtype=torch.long, device=device)


def batch_generate_walks(adj_list: Dict[int, List[int]], 
                         center_nodes: List[int], 
                         n: int, k: int, 
                         device: torch.device) -> torch.Tensor:
    """
    【新函数】批量生成随机游走路径
    Args:
        adj_list: 预构建的邻接表
        center_nodes: 需要游走的中心节点列表
        n: 每个节点游走次数
        k: 步数
    Returns:
        walks: [Num_Samples, N, K+1] 的长整型张量
    """
    all_walks = []
    
    # 纯 Python 循环生成路径（此时因为没有构建邻接表的开销，速度非常快）
    for node in center_nodes:
        node_walks = []
        for _ in range(n):
            w = k_step_random_walk(adj_list, int(node), k)
            node_walks.append(w)
        all_walks.append(node_walks)
    
    return torch.tensor(all_walks, dtype=torch.long, device=device)


def batch_random_walks_for_all_nodes(edge_index: torch.Tensor, n: int, k: int, 
                                     num_nodes: int, device: torch.device = None) -> torch.Tensor:
    """
    为图中所有节点执行N次K步随机游走
    
    Args:
        edge_index: [2, E] 边索引
        n: 每个节点的随机游走次数
        k: 每次随机游走的步数
        num_nodes: 图中节点总数
        device: 设备
    
    Returns:
        all_walks: [num_nodes, N, K+1] 张量，所有节点的随机游走路径
    """
    if device is None:
        device = edge_index.device
    
    all_walks = []
    for node_id in range(num_nodes):
        node_walks = n_times_k_step_random_walks(edge_index, node_id, n, k, num_nodes, device)
        all_walks.append(node_walks)
    
    return torch.stack(all_walks, dim=0)  # [num_nodes, N, K+1]


def get_random_walk_nodes(walks: torch.Tensor, exclude_start: bool = True) -> torch.Tensor:
    """
    从随机游走路径中提取访问的节点（可选择是否排除起始节点）
    
    Args:
        walks: [N, K+1] 或 [num_nodes, N, K+1] 随机游走路径
        exclude_start: 是否排除起始节点
    
    Returns:
        nodes: 提取的节点张量
    """
    if exclude_start:
        if walks.dim() == 2:  # [N, K+1]
            return walks[:, 1:]  # [N, K]
        elif walks.dim() == 3:  # [num_nodes, N, K+1]
            return walks[:, :, 1:]  # [num_nodes, N, K]
    else:
        return walks


def sample_random_walk_subgraph(edge_index: torch.Tensor, center_node: int, 
                               n: int, k: int, num_nodes: int,
                               device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于随机游走采样子图
    
    Args:
        edge_index: [2, E] 边索引
        center_node: 中心节点
        n: 随机游走次数
        k: 随机游走步数
        num_nodes: 图中节点总数
        device: 设备
    
    Returns:
        subgraph_nodes: 子图中的节点
        subgraph_edge_index: 子图的边索引
    """
    if device is None:
        device = edge_index.device
    
    # 执行随机游走
    walks = n_times_k_step_random_walks(edge_index, center_node, n, k, num_nodes, device)
    
    # 收集所有访问过的节点
    visited_nodes = set()
    for walk in walks:
        for node in walk:
            visited_nodes.add(node.item())
    
    subgraph_nodes = torch.tensor(list(visited_nodes), dtype=torch.long, device=device)
    
    # 构建节点映射
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(subgraph_nodes)}
    
    # 提取子图边
    edge_index_cpu = edge_index.cpu().numpy()
    subgraph_edges = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index_cpu[0, i], edge_index_cpu[1, i]
        if src in node_mapping and dst in node_mapping:
            new_src = node_mapping[src]
            new_dst = node_mapping[dst]
            subgraph_edges.append([new_src, new_dst])
    
    if len(subgraph_edges) > 0:
        subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long, device=device).t()
    else:
        subgraph_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return subgraph_nodes, subgraph_edge_index


def compute_random_walk_statistics(walks: torch.Tensor, num_nodes: int) -> dict:
    """
    计算随机游走的统计信息
    
    Args:
        walks: [N, K+1] 或 [num_nodes, N, K+1] 随机游走路径
        num_nodes: 图中节点总数
    
    Returns:
        stats: 包含统计信息的字典
    """
    stats = {}
    
    if walks.dim() == 2:  # 单个节点的随机游走
        n, k_plus_1 = walks.shape
        k = k_plus_1 - 1
        
        # 计算访问频率
        visit_counts = torch.zeros(num_nodes, dtype=torch.float)
        for walk in walks:
            unique_nodes, counts = torch.unique(walk, return_counts=True)
            visit_counts[unique_nodes] += counts.float()
        
        stats['visit_frequency'] = visit_counts / (n * (k + 1))
        stats['unique_nodes_visited'] = (visit_counts > 0).sum().item()
        stats['avg_path_length'] = k + 1
        
    elif walks.dim() == 3:  # 所有节点的随机游走
        num_nodes_batch, n, k_plus_1 = walks.shape
        k = k_plus_1 - 1
        
        # 为每个节点计算统计信息
        node_stats = []
        for node_id in range(num_nodes_batch):
            node_walks = walks[node_id]
            node_stat = compute_random_walk_statistics(node_walks, num_nodes)
            node_stats.append(node_stat)
        
        stats['per_node_stats'] = node_stats
        stats['total_walks'] = num_nodes_batch * n
        stats['walk_length'] = k + 1
    
    return stats

class SpatialPeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # 告诉 PyG：在 Batch 时，spatial_pe_idx 里的索引需要加上之前图的节点数
        if key == 'spatial_pe_idx':
            return self.num_nodes
        if key == 'core_atom_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # 告诉 PyG：spatial_pe_idx 应该在维度 1 上拼接 ([2, N] -> [2, Sum_N])
        if key == 'spatial_pe_idx':
            return 1
        # spatial_pe_val 应该在维度 0 上拼接 ([N] -> [Sum_N])
        if key == 'spatial_pe_val':
            return 0
        if key == 'core_atom_index':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

class AddShortestPathDistance(BaseTransform):
    def __init__(self, cutoff=20, inf_val=100):
        self.cutoff = cutoff
        self.inf_val = inf_val

    def __call__(self, data):
        # 将普通 Data 转换为我们自定义的 SpatialPeData
        # 这样 DataLoader 就能识别 __inc__ 和 __cat_dim__ 了
        new_data = SpatialPeData.from_dict(data.to_dict())
        
        G = to_networkx(new_data, to_undirected=True)
        try:
            dist_dict = dict(nx.all_pairs_shortest_path_length(G))
        except:
            dist_dict = {}

        num_nodes = new_data.num_nodes
        rows = []
        cols = []
        dists = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                rows.append(i)
                cols.append(j)
                d = dist_dict.get(i, {}).get(j, self.inf_val)
                if d > self.cutoff: 
                    d = self.inf_val 
                dists.append(d)

        new_data.spatial_pe_idx = torch.tensor([rows, cols], dtype=torch.long)
        new_data.spatial_pe_val = torch.tensor(dists, dtype=torch.float)
        
        if hasattr(new_data, 'spatial_pe'):
            del new_data.spatial_pe
            
        return new_data


class AddLaplacianPE(BaseTransform):
    def __init__(self, k, attr_name='laplacian_pe', is_undirected=True):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected

    def __call__(self, data):
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # 1. 计算拉普拉斯矩阵
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, 
            normalization='sym', 
            num_nodes=num_nodes
        )
        
        # 转为稀疏矩阵
        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes=num_nodes)

        # 2. 特征分解 (分情况讨论)
        
        # === 情况 A: 图很小，节点数 <= 需要的PE维度 ===
        # 稀疏求解器无法计算所有特征值 (k must be < N)，所以必须转稠密
        if num_nodes <= self.k:
            L_dense = L.toarray() # 转为普通矩阵
            # eigh 可以计算所有特征值
            eig_vals, eig_vecs = la.eigh(L_dense) 
            # 这里的 eig_vecs 形状是 [N, N]
            
            pe = torch.from_numpy(eig_vecs).float()
            
            # 因为 N <= k，我们需要补零到 k 维
            # 比如 N=6, k=8，我们有6个向量，需要补2列0
            pad_dim = self.k - num_nodes
            if pad_dim > 0:
                pe = torch.cat([pe, torch.zeros((num_nodes, pad_dim))], dim=1)

        # === 情况 B: 图够大 ===
        else:
            # 使用稀疏求解器，只算前 k 个
            try:
                # 注意：这里直接取 k 个即可，不需要 k+1 再切片，
                # 除非你明确想跳过最小的那个(特征值为0)。
                # 通常保留所有算出来的 k 个低频分量即可。
                eig_vals, eig_vecs = sla.eigsh(L, k=self.k, which='SM', return_eigenvectors=True)
                
                # 排序 (ARPACK 不保证顺序)
                idx = eig_vals.argsort()
                eig_vals = eig_vals[idx]
                eig_vecs = eig_vecs[:, idx]
                
                pe = torch.from_numpy(eig_vecs).float()
                
            except RuntimeError:
                # 极少数情况 ARPACK 不收敛，兜底方案
                L_dense = L.toarray()
                eig_vals, eig_vecs = la.eigh(L_dense)
                pe = torch.from_numpy(eig_vecs[:, :self.k]).float()

        # 3. 符号翻转 (Sign Flipping) 提示
        # 注意：这里只负责生成数据。在模型 forward 时记得做随机符号翻转。
        
        setattr(data, self.attr_name, pe)
        return data


class AddTopologicalCore(BaseTransform):
    """
    计算图的拓扑中心（核心原子），并存储为 data.core_atom_index
    """
    def __init__(self, method='closeness'):
        self.method = method

    def __call__(self, data):
        # 1. 转换为 NetworkX 图 (无向)
        # to_networkx 默认不拷贝特征，速度较快
        G = to_networkx(data, to_undirected=True)
        
        # 2. 计算中心性
        if self.method == 'closeness':
            centrality = nx.closeness_centrality(G)
        elif self.method == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif self.method == 'degree':
            centrality = nx.degree_centrality(G)
        else:
            # 默认兜底：度中心性 (最快)
            centrality = nx.degree_centrality(G)
        
        # 3. 找到得分最高的节点 ID
        if len(centrality) > 0:
            core_idx = max(centrality, key=centrality.get)
        else:
            core_idx = 0 # 防止空图报错
            
        # 4. 存储到 Data 对象
        # 注意：必须存为 Tensor，且维度通常为 [1] 或 [N] 才能被 PyG 自动处理
        data.core_atom_index = torch.tensor([core_idx], dtype=torch.long)
        
        return data

class AddBRICSFragments(BaseTransform):
    """
    使用 BRICS 算法分解分子，为每个原子分配局部 Fragment ID。
    """
    def __call__(self, data):
        if Chem is None or BRICS is None:
            raise ModuleNotFoundError("rdkit is required for AddBRICSFragments, but it's not installed.")
        # 1. 获取 Mol 对象 (假设 dataset 里有 smiles，或者 data.smiles)
        # OGB 数据集通常在 data object 里不带 smiles，需要从 raw csv 读取注入
        # 这里假设 data.smiles 已经存在 (参考之前的注入逻辑)
        if not hasattr(data, 'smiles'):
            # 兜底：如果没有 SMILES，整个分子算作一个 Fragment
            data.fragment_index = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
            
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            data.fragment_index = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        # 2. 寻找 BRICS 可断裂键
        # FindBRICSBonds 返回的是 tuple ((atom1, atom2), (type1, type2))
        res = list(BRICS.FindBRICSBonds(mol))
        bonds_to_break = [b[0] for b in res] # 只取原子索引对

        # 3. 构建图并断键
        # 使用 NetworkX 来找连通分量，比直接操作 Mol 更方便
        G = nx.Graph()
        G.add_nodes_from(range(mol.GetNumAtoms()))
        
        # 添加所有键
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            G.add_edge(u, v)
        
        # 移除 BRICS 键
        for (u, v) in bonds_to_break:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        
        # 4. 寻找连通分量 (即 Fragments)
        # fragments 是一个 list of sets: [{0,1,2,3,4,5}, {6,7}, ...]
        fragments = list(nx.connected_components(G))
        
        # 5. 生成 Fragment Index
        # 给每个原子打上它是第几个 Fragment 的标签
        frag_idx = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, frag_atoms in enumerate(fragments):
            for atom_id in frag_atoms:
                # 注意：这里的 atom_id 必须和 PyG 的 x 顺序一致
                # 通常 RDKit 读取顺序和 OGB 提供的 x 顺序是一致的
                if atom_id < data.num_nodes:
                    frag_idx[atom_id] = i
                    
        data.fragment_index = frag_idx
        return data
    
import warnings
class AddSpectralClusters(BaseTransform):
    """
    通用图划分：基于谱聚类将图划分为 K 个子图。
    支持【固定 K 值】与【大图自适应 K 值】两种模式。
    """
    def __init__(self, adaptive=False, n_clusters=3, min_k=5, ratio=20):
        """
        :param adaptive: 是否开启自适应 K 模式 (针对社交网络大图)
        :param n_clusters: 固定模式下的 K 值 (针对分子图)
        :param min_k: 自适应模式下的最小子图数
        :param ratio: 自适应模式下的节点比例 (K = N // ratio)
        """
        self.adaptive = adaptive
        self.n_clusters = n_clusters
        self.min_k = min_k
        self.ratio = ratio
        
        # 忽略 KMeans 的内存泄漏警告（在 Windows 上常见，不影响训练）
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster")

    def __call__(self, data):
        num_nodes = data.num_nodes
        
        # 1. 动态计算当前图的 K 值
        if self.adaptive:
            target_k = max(self.min_k, num_nodes // self.ratio)
        else:
            target_k = self.n_clusters
            
        # 【核心安全锁】：eigsh 要求提取的特征向量个数 k 必须严格小于矩阵维度 N
        # 如果算出的 K 已经超过或等于节点数，强制让 K = num_nodes - 1
        current_k = min(target_k, num_nodes - 1)

        # 2. 极小图处理：如果 K 退化到 1 甚至 0（例如全图只有1或2个节点）
        if current_k <= 1:
            data.fragment_index = torch.zeros(num_nodes, dtype=torch.long)
            return data

        # 3. 计算拉普拉斯矩阵
        edge_index, edge_weight = get_laplacian(
            data.edge_index, 
            normalization='sym', 
            num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)
        
        # 4. 求解特征向量
        try:
            # 'SM' = Smallest Magnitude
            # 使用 current_k 保证严格小于 num_nodes
            eig_vals, eig_vecs = sla.eigsh(L, k=current_k, which='SM')
        except Exception as e:
            # 兜底：如果 eigsh 依然因为矩阵奇异等原因失败，退化为单一聚类
            data.fragment_index = torch.zeros(num_nodes, dtype=torch.long)
            return data
            
        # 5. K-Means 聚类
        # 将 n_init 设置为 10（或者 'auto'）可以加快预处理速度
        kmeans = KMeans(n_clusters=current_k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(eig_vecs)
        
        data.fragment_index = torch.from_numpy(labels).long()
        
        return data
# class AddSpectralClusters(BaseTransform):
#     """
#     通用图划分：基于谱聚类将图划分为 K 个子图
#     """
#     def __init__(self, n_clusters=3):
#         self.n_clusters = n_clusters

#     def __call__(self, data):
#         num_nodes = data.num_nodes
        
#         # 1. 极小图处理：如果节点数少于聚类数，直接全归为一类
#         if num_nodes <= self.n_clusters:
#             data.fragment_index = torch.zeros(num_nodes, dtype=torch.long)
#             return data

#         # 2. 计算拉普拉斯位置编码 (LapPE) 作为聚类特征
#         # 谱聚类的本质就是对 LapPE 进行 K-Means
#         edge_index, edge_weight = get_laplacian(
#             data.edge_index, 
#             normalization='sym', 
#             num_nodes=num_nodes
#         )
#         L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)
        
#         # 计算前 k 个特征向量 (k = n_clusters)
#         import scipy.sparse.linalg as sla
#         try:
#             # 'SM' = Smallest Magnitude
#             eig_vals, eig_vecs = sla.eigsh(L, k=self.n_clusters, which='SM')
#         except:
#             # 兜底：如果 eigsh 失败，用随机划分或全0
#             data.fragment_index = torch.zeros(num_nodes, dtype=torch.long)
#             return data
            
#         # 3. K-Means 聚类
#         # eig_vecs shape: [N, K]
#         kmeans = KMeans(n_clusters=self.n_clusters,random_state=42, n_init=20)
#         labels = kmeans.fit_predict(eig_vecs)
        
#         data.fragment_index = torch.from_numpy(labels).long()
        
#         return data
