import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from kd_loss import AddLaplacianPE, AddShortestPathDistance, AddTopologicalCore, AddSpectralClusters

class ProcessedDataset(InMemoryDataset):
    """用于存储经过谱聚类处理后的数据"""
    def __init__(self, data_list, slices=None):
        super().__init__(None, None, None)
        self.data, self.slices = self.collate(data_list)

def load_zinc_dataset(root='./ZINC', batch_size=128, model_type='GIN',
                      pe_dim=None, 
                      add_spd=False,
                      add_core=True,
                      add_spectral=True,       # 控制是否开启子图划分
                      n_clusters=5,            # 子图数量 (超参数)
                      force_reload=False):     # 新增 force_reload 参数
    
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
        transforms_list.append(AddSpectralClusters(n_clusters=n_clusters))
        
    if len(transforms_list) > 0:
        base_transform = Compose(transforms_list)
    else:
        base_transform = None
    
    # 加载数据集
    # 注意：ZINC 数据集如果 force_reload=True 会重新处理数据
    train_dataset = ZINC(root=root, subset=True, split='train', pre_transform=base_transform, force_reload=force_reload)
    val_dataset = ZINC(root=root, subset=True, split='val', pre_transform=base_transform, force_reload=force_reload)
    test_dataset = ZINC(root=root, subset=True, split='test', pre_transform=base_transform, force_reload=force_reload)

    # # 3. 执行谱聚类 (如果开启)
    # if add_spectral:
    #     print(f"正在执行谱聚类 (Spectral Clustering), k={n_clusters} ...")
    #     spectral_transform = AddSpectralClusters(n_clusters=n_clusters)
        
    #     def process_split(dataset, name):
    #         print(f"Processing {name} split...")
    #         data_list = []
    #         for i in tqdm(range(len(dataset))):
    #             data = dataset[i]
    #             # 执行聚类，生成 fragment_index
    #             data = spectral_transform(data)
    #             data_list.append(data)
    #         return ProcessedDataset(data_list)

    #     train_dataset = process_split(train_dataset, 'train')
    #     val_dataset = process_split(val_dataset, 'val')
    #     test_dataset = process_split(test_dataset, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data = train_dataset[0]
    dataset_info = {
        'num_graphs': len(train_dataset) + len(val_dataset) + len(test_dataset),
        'num_classes': 1,
        # 【关键】这里不再是特征维度，而是词表大小 (Vocab Size)
        # ZINC 实际上最大原子索引是 27，所以设 28 安全，或者 train_dataset.num_node_features
        # PyG ZINC num_node_features 通常也是 21 或 28
        'num_features': 28, 
        'edge_attr': True,
        'edge_attr_dim': 0 
    }
    
    return train_loader, val_loader, test_loader, dataset_info

if __name__ == '__main__':
    train_loader, val_loader, test_loader, dataset_info = load_zinc_dataset(
        root='./ZINC', 
        batch_size=128,
        model_type='GIN',
        pe_dim=8,
        add_spd=False,
        add_core=True,
        add_spectral=True,
    )