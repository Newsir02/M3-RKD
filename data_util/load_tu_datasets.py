import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose

# 假设你的自定义 Transform 类都在 utils 包里
# 如果在当前文件，请直接使用；如果在其他文件，请确保 import 路径正确
from kd_loss import AddLaplacianPE , AddShortestPathDistance, AddTopologicalCore, AddSpectralClusters

def load_tu_dataset(
    name: str,
    batch_size: int,
    root: str = './TUdatasets',
):
    """
    加载 TUDataset 数据集，包含高级图结构预处理（PE, Spectral, Core等）。
    同时保持固定的 Train/Val/Test 划分。
    """

    dataset = TUDataset(
        root=root, 
        name=name,
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