
from torch_geometric.datasets import TUDataset
import torch

print('=== 推荐的小型多图数据集 ===\n')

# 定义几个适合方法验证的小型数据集
datasets_info = [
    #('MUTAG', '化学分子数据集，二分类'),
    #('ENZYMES', '蛋白质数据集，6分类'),
    ('PROTEINS', '蛋白质数据集，二分类'),
    #('IMDB-BINARY', '社交网络数据集，二分类'),
    #('REDDIT-BINARY', '社交网络数据集，二分类'),
    #('COLLAB', '科学合作网络，三分类')
]

for name, description in datasets_info:
    try:
        dataset = TUDataset(root=f'./TUDatasets/{name}', name=name)
        print(f'数据集: {name}')
        print(f'描述: {description}')
        print(f'图数量: {len(dataset)}')
        print(f'类别数: {dataset.num_classes}')
        
        # 获取第一个图的信息
        data = dataset[0]
        print(f'平均节点数: ~{data.num_nodes} (第一个图)')
        print(f'平均边数: ~{data.num_edges} (第一个图)')
        if hasattr(data, 'x') and data.x is not None:
            print(f'节点特征维度: {data.x.shape[1]}')
        else:
            print('节点特征维度: 无节点特征')
        print('-' * 50)
        
    except Exception as e:
        print(f'数据集 {name} 加载失败: {e}')
        print('-' * 50)
