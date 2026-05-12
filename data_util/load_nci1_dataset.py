import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose

# 假设你的自定义 Transform 类都在 utils 包里
# 如果在当前文件，请直接使用；如果在其他文件，请确保 import 路径正确
from kd_loss import AddLaplacianPE , AddShortestPathDistance, AddTopologicalCore, AddSpectralClusters

def load_nci1_dataset(
    batch_size: int, 
    use_degree: bool = True,
    # --- 新增的预处理参数 ---
    pe_dim: int = 0,
    add_spd: bool = False,
    add_core: bool = False,
    add_spectral: bool = False,
    n_clusters: int = 5,
    name: str = 'NCI1'
):
    """
    加载 NCI1 数据集，包含高级图结构预处理（PE, Spectral, Core等）。
    同时保持固定的 Train/Val/Test 划分。
    """
    transforms_list = []

    if use_degree:
        transforms_list.append(T.OneHotDegree(max_degree=100))
    if pe_dim > 0:
        transforms_list.append(AddLaplacianPE(k=pe_dim, attr_name='laplacian_pe'))
    if add_spd:
        transforms_list.append(AddShortestPathDistance(cutoff=20, inf_val=100))
    if add_core:
        # Closeness Centrality 也是通用的拓扑指标
        transforms_list.append(AddTopologicalCore(method='closeness')) 
    if add_spectral:
        transforms_list.append(AddSpectralClusters(n_clusters=n_clusters))

    # 组合所有变换
    pre_transform = Compose(transforms_list) if len(transforms_list) > 0 else None

    print(f"正在加载 {name}数据集...")
    print(f"  - Transforms: OneHot={use_degree}, PE={pe_dim}, SPD={add_spd}, Core={add_core}, Spectral={add_spectral}")

    dataset = TUDataset(
        root='./TUdatasets', 
        name=name,
        pre_transform=pre_transform,
        use_node_attr=True  # NCI1 有原始特征
    )

    DATA_SPLIT_SEED = 42 
    
    # 保存当前模型随机状态 (由外部 seed 控制)
    current_rng_state = torch.get_rng_state()
    
    # 切换到数据划分种子
    torch.manual_seed(DATA_SPLIT_SEED)
    
    dataset = dataset.shuffle()
    
    torch.set_rng_state(current_rng_state)
    
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train+n_val]
    test_dataset = dataset[n_train+n_val:]

    print(f"  [Check] 第一个训练样本 Label: {train_dataset[0].y.item()}")
    print(f"  [Check] 第一个测试样本 Label: {test_dataset[0].y.item()}")

    # ==========================================
    # 4. DataLoader
    # ==========================================
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataset_info = {
        'num_features': dataset.num_features,
        'num_classes': dataset.num_classes,
        'num_graphs': len(dataset)
    }
    
    return train_loader, val_loader, test_loader, dataset_info