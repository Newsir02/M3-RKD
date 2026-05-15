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

    row, col = edge_index.cpu().numpy()

    data = np.ones(len(row), dtype=np.int32)
    adj_mat = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    adj_lil = adj_mat.tolil()

    return adj_lil.rows 

def k_step_random_walk(adj_list, start_node: int, k: int) -> List[int]:
    walk_path = [start_node]
    current_node = start_node

    for step in range(k):

        if isinstance(adj_list, dict):
            neighbors = adj_list.get(current_node, [])

        else:
            neighbors = adj_list[current_node]

        if not neighbors:        
            walk_path.append(current_node)
        else:
            prev_node = walk_path[-2] if len(walk_path) > 1 else None
            if prev_node is not None and len(neighbors) > 1:

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

    return torch.stack(all_walks, dim=0)                       

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
        if walks.dim() == 2:            
            return walks[:, 1:]          
        elif walks.dim() == 3:                       
            return walks[:, :, 1:]                     
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

    walks = n_times_k_step_random_walks(edge_index, center_node, n, k, num_nodes, device)

    visited_nodes = set()
    for walk in walks:
        for node in walk:
            visited_nodes.add(node.item())

    subgraph_nodes = torch.tensor(list(visited_nodes), dtype=torch.long, device=device)

    node_mapping = {old_id: new_id for new_id, old_id in enumerate(subgraph_nodes)}

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

    if walks.dim() == 2:             
        n, k_plus_1 = walks.shape
        k = k_plus_1 - 1

        visit_counts = torch.zeros(num_nodes, dtype=torch.float)
        for walk in walks:
            unique_nodes, counts = torch.unique(walk, return_counts=True)
            visit_counts[unique_nodes] += counts.float()

        stats['visit_frequency'] = visit_counts / (n * (k + 1))
        stats['unique_nodes_visited'] = (visit_counts > 0).sum().item()
        stats['avg_path_length'] = k + 1

    elif walks.dim() == 3:             
        num_nodes_batch, n, k_plus_1 = walks.shape
        k = k_plus_1 - 1

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

        if key == 'spatial_pe_idx':
            return self.num_nodes
        if key == 'core_atom_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):

        if key == 'spatial_pe_idx':
            return 1

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

        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, 
            normalization='sym', 
            num_nodes=num_nodes
        )

        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes=num_nodes)

        if num_nodes <= self.k:
            L_dense = L.toarray()         

            eig_vals, eig_vecs = la.eigh(L_dense) 

            pe = torch.from_numpy(eig_vecs).float()

            pad_dim = self.k - num_nodes
            if pad_dim > 0:
                pe = torch.cat([pe, torch.zeros((num_nodes, pad_dim))], dim=1)

        else:

            try:

                eig_vals, eig_vecs = sla.eigsh(L, k=self.k, which='SM', return_eigenvectors=True)

                idx = eig_vals.argsort()
                eig_vals = eig_vals[idx]
                eig_vecs = eig_vecs[:, idx]

                pe = torch.from_numpy(eig_vecs).float()

            except RuntimeError:

                L_dense = L.toarray()
                eig_vals, eig_vecs = la.eigh(L_dense)
                pe = torch.from_numpy(eig_vecs[:, :self.k]).float()

        setattr(data, self.attr_name, pe)
        return data

class AddTopologicalCore(BaseTransform):
    """
    计算图的拓扑中心（核心原子），并存储为 data.core_atom_index
    """
    def __init__(self, method='closeness'):
        self.method = method

    def __call__(self, data):

        G = to_networkx(data, to_undirected=True)

        if self.method == 'closeness':
            centrality = nx.closeness_centrality(G)
        elif self.method == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif self.method == 'degree':
            centrality = nx.degree_centrality(G)
        else:

            centrality = nx.degree_centrality(G)

        if len(centrality) > 0:
            core_idx = max(centrality, key=centrality.get)
        else:
            core_idx = 0         

        data.core_atom_index = torch.tensor([core_idx], dtype=torch.long)

        return data

class AddBRICSFragments(BaseTransform):
    """
    使用 BRICS 算法分解分子，为每个原子分配局部 Fragment ID。
    """
    def __call__(self, data):
        if Chem is None or BRICS is None:
            raise ModuleNotFoundError("rdkit is required for AddBRICSFragments, but it's not installed.")

        if not hasattr(data, 'smiles'):

            data.fragment_index = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            data.fragment_index = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        res = list(BRICS.FindBRICSBonds(mol))
        bonds_to_break = [b[0] for b in res]          

        G = nx.Graph()
        G.add_nodes_from(range(mol.GetNumAtoms()))

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            G.add_edge(u, v)

        for (u, v) in bonds_to_break:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

        fragments = list(nx.connected_components(G))

        frag_idx = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, frag_atoms in enumerate(fragments):
            for atom_id in frag_atoms:

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

        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster")

    def __call__(self, data):
        num_nodes = data.num_nodes

        if self.adaptive:
            target_k = max(self.min_k, num_nodes // self.ratio)
        else:
            target_k = self.n_clusters

        current_k = min(target_k, num_nodes - 1)

        if current_k <= 1:
            data.fragment_index = torch.zeros(num_nodes, dtype=torch.long)
            return data

        edge_index, edge_weight = get_laplacian(
            data.edge_index, 
            normalization='sym', 
            num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

        try:

            eig_vals, eig_vecs = sla.eigsh(L, k=current_k, which='SM')
        except Exception as e:

            data.fragment_index = torch.zeros(num_nodes, dtype=torch.long)
            return data

        kmeans = KMeans(n_clusters=current_k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(eig_vecs)

        data.fragment_index = torch.from_numpy(labels).long()

        return data
