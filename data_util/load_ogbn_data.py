import torch
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import os
import pandas as pd
from kd_loss import AddLaplacianPE, AddShortestPathDistance, AddTopologicalCore, AddBRICSFragments,AddSpectralClusters
def load_ogbn_arxiv(dataset_name):
    """简单加载OGBN-Arxiv节点分类数据集"""
    dataset = PygNodePropPredDataset(name=dataset_name, root='./dataset', transform=ToUndirected(),pe_dim = None)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    return data, split_idx

def load_ogbg_dataset(root: str = './OGBdataset', 
                      batch_size: int = 128, 
                      name: str = 'ogbg-molhiv', 
                      pe_dim = None, 
                      add_spd = False,
                      add_core = True,
                      # --- 修改这里：只保留谱聚类控制 ---
                      add_spectral = True,       # 控制是否开启子图划分
                      n_clusters = 5,            # 子图数量 (超参数)
                      # -------------------------------
                      subset_ratio: float = 1.0,
                      force_reload: bool = False
                      ):
    
    # 1. 基础 Transforms (PE, SPD, Core, Spectral)
    transforms_list = []
    if pe_dim:
        transforms_list.append(AddLaplacianPE(k=pe_dim, attr_name='laplacian_pe'))
    if add_spd:
        transforms_list.append(AddShortestPathDistance(cutoff=20, inf_val=100))
    if add_core:
        # Closeness Centrality 也是通用的拓扑指标
        transforms_list.append(AddTopologicalCore(method='closeness')) 
    if add_spectral:
        transforms_list.append(AddSpectralClusters(adaptive=False, n_clusters=n_clusters))
        
    if len(transforms_list) > 0:
        base_transform = Compose(transforms_list)
    else:
        base_transform = None

    # 2. 加载原始数据集
    print(f"Loading dataset: {name}...")
    # 判断是否为 OGB 数据集
    if name.startswith('ogbg-'):
        # PygGraphPropPredDataset 不直接支持 force_reload，但可以通过 pre_transform 实现缓存
        dataset = PygGraphPropPredDataset(name=name, root=root, pre_transform=base_transform)
    else:
        # 支持 PROTEINS, IMDB 等通用数据集
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root=root, name=name, pre_transform=base_transform, force_reload=force_reload)

    # 3. 获取划分索引
    if hasattr(dataset, 'get_idx_split'):
        split_idx = dataset.get_idx_split()
    else:
        # 对于没有预定义划分的 TUDataset，手动随机划分 (8:1:1)
        num_graphs = len(dataset)
        # 使用固定种子保证可复现
        g = torch.Generator()
        g.manual_seed(42)
        perm = torch.randperm(num_graphs, generator=g)
        split_idx = {
            'train': perm[:int(0.8*num_graphs)],
            'valid': perm[int(0.8*num_graphs):int(0.9*num_graphs)],
            'test': perm[int(0.9*num_graphs):]
        }

    # 4. 数据集截断 (Subset)
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    if subset_ratio < 1.0:
        print(f"⚠️ 截断数据集: {subset_ratio*100}%")
        n_train = int(len(train_idx) * subset_ratio)
        n_valid = int(len(valid_idx) * subset_ratio)
        n_test  = int(len(test_idx)  * subset_ratio)
        train_idx = train_idx[:n_train]
        valid_idx = valid_idx[:n_valid]
        test_idx  = test_idx[:n_test]

    # 5. DataLoader
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[valid_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)

    # 6. Info
    data0 = dataset[0]
    num_tasks = dataset.num_tasks if hasattr(dataset, 'num_tasks') else dataset.num_classes
    
    dataset_info = {
        'name': name,
        'num_graphs': len(dataset),
        'num_features': data0.x.shape[1],
        'num_tasks': num_tasks,
        'edge_attr': hasattr(data0, 'edge_attr') and data0.edge_attr is not None,
        'edge_attr_dim': data0.edge_attr.shape[1] if (hasattr(data0, 'edge_attr') and data0.edge_attr is not None) else 0,
        'avg_nodes': float(sum([d.num_nodes for d in dataset]) / len(dataset)),
    }

    return train_loader, val_loader, test_loader, dataset_info


