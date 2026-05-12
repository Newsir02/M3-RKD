from sympy.logic import false
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToUndirected, OneHotDegree
from torch_geometric.data import InMemoryDataset
import torch
from sklearn.model_selection import train_test_split

from kd_loss import AddLaplacianPE , AddShortestPathDistance, AddTopologicalCore, AddSpectralClusters
from tqdm import tqdm

class ProcessedDataset(InMemoryDataset):
    """
    用于封装处理后（如添加了子图索引）数据的通用内存数据集类
    """
    def __init__(self, data_list, num_classes=None, num_node_features=None):
        super().__init__(None, None, None)
        self.data, self.slices = self.collate(data_list)
        self._num_classes = num_classes
        self._num_node_features = num_node_features

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_node_features(self):
        return self._num_node_features
def load_multi_graph_dataset(dataset_name, root='./TUdatasets', batch_size=32, 
                             test_size=0.2, val_size=0.1, random_state=0,
                             pe_dim=None, add_spd=False, add_core=True,
                             # --- 新增参数 ---
                             add_spectral=False,  # 是否开启谱聚类
                             n_clusters=3 ,
                             force_reload=False       
                             ):

    # transforms_list = [OneHotDegree(max_degree=135)]
    transforms_list = []
    if pe_dim:
        transforms_list.append(AddLaplacianPE(k=pe_dim, attr_name='laplacian_pe', is_undirected=True))
    if add_spd:
        transforms_list.append(AddShortestPathDistance(cutoff=20, inf_val=100))
    if add_core:
        transforms_list.append(AddTopologicalCore(method='closeness'))
    if add_spectral:
        transforms_list.append(AddSpectralClusters(n_clusters=n_clusters))
    
    pre_transform = Compose(transforms_list) if len(transforms_list) > 0 else None

    # 2. 加载原始数据集
    print(f"Loading {dataset_name}...")
    dataset = TUDataset(
        root=f'{root}', 
        name=dataset_name,
        transform=ToUndirected(),
        use_node_attr=True,
        pre_transform=pre_transform,
        force_reload=force_reload
    )

    # 4. 提取数据集统计信息 (使用 dataset[0] 前要确保非空)
    data0 = dataset[0]
    dataset_info = {
        'name': dataset_name,
        'num_graphs': len(dataset),
        'num_classes': dataset.num_classes,
        'num_features': dataset.num_node_features,
        'pe_dim': pe_dim if pe_dim else None,
        # 计算平均值
        'avg_nodes': float(sum([d.num_nodes for d in dataset]) / len(dataset)),
        'avg_edges': float(sum([d.num_edges for d in dataset]) / len(dataset)),
        # TUDataset 的 edge_attr 处理
        'edge_attr': hasattr(data0, 'edge_attr') and data0.edge_attr is not None,
        'edge_attr_dim': data0.edge_attr.shape[1] if (hasattr(data0, 'edge_attr') and data0.edge_attr is not None) else 0
    }
    
    # 5. 数据集划分 (Stratified Split)
    indices = list(range(len(dataset)))
    labels = [data.y.item() for data in dataset] # 获取所有标签用于分层
    
    # Train / (Val + Test)
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, labels, test_size=(test_size + val_size), 
        random_state=random_state, stratify=labels
    )
    
    # Val / Test (在剩余数据中分)
    # 计算 Val 在 Temp 中的比例: val / (val + test)
    val_rel_size = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_rel_size), 
        random_state=random_state, stratify=temp_labels
    )
    
    # 6. 创建 DataLoader
    # 优化：直接使用 dataset 切片，保持 PyG Dataset 对象特性
    # 将 numpy/list 类型的 indices 转换为 torch.tensor 或直接作为 list 索引
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset_info
