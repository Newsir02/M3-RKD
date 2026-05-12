import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose
from kd_loss import AddLaplacianPE , AddShortestPathDistance, AddTopologicalCore, AddSpectralClusters
from torch_geometric.utils import degree
import torch.nn.functional as F

class ClampedOneHotDegree(object):
    """
    带有截断功能的 OneHot 度数编码。
    所有大于 max_degree 的度数，都会被强行归入 max_degree 这一类别中。
    """
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, data):
        idx, x = data.edge_index[0], data.x
        # 计算每个节点的度数
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        # 🔴 核心魔法：大于 max_degree 的全部按 max_degree 算
        deg = deg.clamp(max=self.max_degree)
        
        # 转换为 One-Hot (维度固定为 max_degree + 1)
        deg_one_hot = F.one_hot(deg, num_classes=self.max_degree + 1).float()
        
        if x is not None:
            data.x = torch.cat([x, deg_one_hot], dim=-1)
        else:
            data.x = deg_one_hot
            
        return data
    
class OnlyDegree(object):
    def __call__(self, data):
        # 计算度数 (N, 1)
        row, col = data.edge_index
        deg = degree(col, data.num_nodes).view(-1, 1)
        data.x = deg.float()
        return data

def load_tu_dataset(
    name: str,
    root: str,
    batch_size: int, 
    use_degree: bool = True,
    pe_dim: int = 0,
    add_spd: bool = False,
    add_core: bool = False,
    add_spectral: bool = False,
    n_clusters: int = 5
):
    """
    Load a generic TU dataset (e.g. IMDB-BINARY, REDDIT-BINARY).
    """
    transforms_list = []

    if use_degree:
        if name in ['IMDB-BINARY']:  
            transforms_list.append(T.OneHotDegree(max_degree=20))
        elif name in ['REDDIT-BINARY']:
            transforms_list.append(ClampedOneHotDegree(max_degree=20))
        elif name in ['COLLAB']:
            transforms_list.append(ClampedOneHotDegree(max_degree=20))
        elif name in ['REDDIT-MULTI-5K']:
            transforms_list.append(ClampedOneHotDegree(max_degree=20))
        else:
            transforms_list.append(T.OneHotDegree(max_degree=100))
    # elif name in ['REDDIT-BINARY', 'IMDB-BINARY']:
    #     # For social datasets without node features, use constant features if degree is not used
    #     transforms_list.append(T.Constant(value=1.0))
    # elif name in ['COLLAB','REDDIT-MULTI-5K']:
    #       transforms_list.append(T.LocalDegreeProfile())
    #       transforms_list.append(T.NormalizeFeatures())
    # elif name in ['REDDIT-MULTI-5K', 'COLLAB']:
    #     # 替代之前的 T.LocalDegreeProfile()
    #     transforms_list.append(OnlyDegree())
    #     # 依然建议加个 Normalize，防止度数太大导致 MLP 梯度爆炸
    #     transforms_list.append(T.NormalizeFeatures())

    # if name in ['PROTEINS','DD','ENZYMES']:
    #     transforms_list.append(T.NormalizeFeatures())

    if pe_dim > 0:
        transforms_list.append(AddLaplacianPE(k=pe_dim, attr_name='laplacian_pe'))
    if add_spd:
        transforms_list.append(AddShortestPathDistance(cutoff=20, inf_val=100))
    if add_core:
        transforms_list.append(AddTopologicalCore(method='closeness')) 
    if add_spectral:
        if name in ['IMDB-BINARY']:            
            transforms_list.append(AddSpectralClusters(adaptive=True, n_clusters=n_clusters, ratio=6))
        elif name in ['REDDIT-BINARY']:
            transforms_list.append(AddSpectralClusters(adaptive=True, n_clusters=n_clusters, ratio=20))
        else:
            transforms_list.append(AddSpectralClusters(adaptive=False, n_clusters=n_clusters))
            
    pre_transform = Compose(transforms_list) if len(transforms_list) > 0 else None

    print(f"Loading {name} dataset from {root}...")
    
    if name in ['Frankenstein', 'PROTEINS', 'ENZYMES']:
        dataset = TUDataset(
            root=root, 
            name=name,
            pre_transform=pre_transform,
            use_node_attr=True,
        )
    else:
        dataset = TUDataset(
            root=root, 
            name=name,
            pre_transform=pre_transform,
            use_node_attr=False,
        )
    return dataset
