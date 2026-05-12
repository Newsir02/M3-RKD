from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected,NormalizeFeatures,Compose

def load_cora_data(dataset_name):
    """加载Cora数据集"""
    dataset = Planetoid(root='./dataset', name="Cora", transform=Compose([ToUndirected(),NormalizeFeatures()]))
    data = dataset[0]

    train_mask = data.train_mask
    valid_mask = data.val_mask
    test_mask = data.test_mask
    
    spilt_mask = {
        'train': train_mask,
        'valid': valid_mask,
        'test': test_mask
    }

    return data, spilt_mask

if __name__ == "__main__":
    dataset_name = 'Cora'
    data, spilt_mask = load_cora_data(dataset_name)
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"特征维度: {data.x.shape[1]}")
    print(f"类别数: {data.y.max().item() + 1}")