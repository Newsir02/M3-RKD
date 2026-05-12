import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import sys
import random
import argparse
import time
from tqdm import tqdm

from model.baseGNN import GIN,GCN,SAGE
from model.MLP import MLP
from data_util.load_tu_dataset import load_tu_dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from kd_loss import (
    nce_criterion, 
    random_walk_distillation_loss, 
    kl_divergence_loss,
    graph_topology_attention_loss, 
    compute_core_view_graph_emb, 
    rkd_distance_loss, 
    subgraph_distillation_loss, 
    topk_ranking_distillation_loss
)

class ProjectionMLP(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.projection_head(x)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            if isinstance(model, (GIN, MLP,GCN,SAGE)):
                logits = model(batch)
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
                
            loss = criterion(logits, batch.y)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            total_loss += loss.item()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'loss': avg_loss}

def train_step(teacher_model, student_model, batch, optimizer, device, config=None, student_proj_head=None):
    teacher_model.eval()
    student_model.train()
    optimizer.zero_grad()

    batch = batch.to(device)
    
    if hasattr(batch, 'core_atom_index'):
        core_global_indices = batch.core_atom_index.view(-1)
    else:
        core_global_indices = None

    if config['kl_weight'] > 0 or config['core_nce_weight'] > 0 or config['subgraph_weight'] > 0 or config['topk_weight'] > 0:
        with torch.no_grad():
            t_logits, t_node_emb, t_graph_emb = teacher_model(batch, True)
            
    s_logits, s_node_emb, s_graph_emb = student_model(batch, True)

    criterion = nn.CrossEntropyLoss()
    task_loss = criterion(s_logits, batch.y)

    kl_loss = torch.tensor(0.0, device=device)
    loss_core_rkd = torch.tensor(0.0, device=device)
    loss_sub_feat = torch.tensor(0.0, device=device)
    loss_sub_rel = torch.tensor(0.0, device=device)
    loss_topk = torch.tensor(0.0, device=device)

    # KL Divergence
    total_kd_loss = torch.tensor(0.0, device=device)
    if config['kl_weight'] > 0:
        kl_loss = kl_divergence_loss(s_logits, t_logits, temperature=config['kl_temperature'])
        total_kd_loss += config['kl_weight'] * kl_loss
        
    if config['core_nce_weight'] > 0 or config['subgraph_weight'] > 0 or config['topk_weight'] > 0:
        # t_proj = teacher_proj_head(t_node_emb)
        # s_proj = student_proj_head(s_node_emb)

        if config['core_nce_weight'] > 0 and core_global_indices is not None:
            teacher_core_emb = compute_core_view_graph_emb(t_node_emb, batch.batch, core_global_indices,t=config['core_temperature'])
            student_core_emb = compute_core_view_graph_emb(s_node_emb, batch.batch, core_global_indices,t=config['core_temperature'])
            loss_core_rkd = rkd_distance_loss(teacher_core_emb.detach(), student_core_emb, delta=1.0)

        if config['subgraph_weight'] > 0 and hasattr(batch, 'fragment_index'):
            sparse_idx = batch.fragment_index
            _, dense_idx = torch.unique(sparse_idx, return_inverse=True)
            global_frag_index = dense_idx
            loss_sub_feat, loss_sub_rel = subgraph_distillation_loss(
                t_node_emb, s_node_emb, global_frag_index, batch.batch, loss_type='kl', temperature=config['sub_temperature']
            )
            
        if config.get('topk_weight', 0) > 0:
            loss_topk = topk_ranking_distillation_loss(
                t_node_emb, s_node_emb, batch.batch, k=config['topk_k'], temperature=config.get('rank_temperature', 0.1)
            )

    total_kd_loss += (config['core_nce_weight'] * loss_core_rkd + 
                      config['subgraph_weight'] * (loss_sub_feat + loss_sub_rel) + config.get('topk_weight', 0) * loss_topk)

    total_loss = config['task_weight'] * task_loss + total_kd_loss
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'kd_loss': total_kd_loss.item() if isinstance(total_kd_loss, torch.Tensor) else total_kd_loss,
        'kl_loss': kl_loss.item(),
        'loss_core_rkd': loss_core_rkd.item(),
        'loss_sub_feat': loss_sub_feat.item(),
        'loss_sub_rel': loss_sub_rel.item(),
        'loss_topk': loss_topk.item()
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # --- 关键修改开始 ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def train_and_evaluate_kd(config, device, base_output_dir, fold, train_loader, val_loader, test_loader, dataset_info):
    seed = fold
    print(f"\n{'='*40}")
    print(f"开始 Fold {fold} 蒸馏训练 | Seed: {seed}")
    print(f"{'='*40}")

    set_seed(seed)
    
    # 构造教师模型
    if config['teacher_model'] == 'GIN':
        teacher_model = GIN(
            num_layers=config['tea_layers'],
            hidden_dim=config['tea_hidden_dim'],
            num_classes=dataset_info['num_classes'],
            input_dim=dataset_info['num_features'],
            edge_attr_dim=dataset_info['edge_attr_dim'] if dataset_info['edge_attr'] else None,
            edge_attr=dataset_info['edge_attr'] if dataset_info['edge_attr'] else None,
            dropout=config['dropout_tea'],
            pooling_method=config['tea_pool_type'],
            useEmbedding=False,
            torchEmbedding=False,
            addVirtualNode=False,
            add_residual=False
        )
    elif config['teacher_model'] == 'GCN':
        teacher_model = GCN(
            num_layers=config['tea_layers'],
            hidden_dim=config['tea_hidden_dim'],
            num_classes=dataset_info['num_classes'],
            input_dim=dataset_info['num_features'],
            edge_attr_dim=dataset_info['edge_attr_dim'] if dataset_info['edge_attr'] else None,
            edge_attr=dataset_info['edge_attr'] if dataset_info['edge_attr'] else None,
            dropout=config['dropout_tea'],
            pooling_method=config['tea_pool_type'],
            useEmbedding=False,
            torchEmbedding=False,
            # addVirtualNode=True,
            # add_residual=False
            addVirtualNode=False,
            add_residual=True
        )
    elif config['teacher_model'] == 'SAGE':
        teacher_model = SAGE(
            num_layers=config['tea_layers'],
            hidden_dim=config['tea_hidden_dim'],
            num_classes=dataset_info['num_classes'],
            input_dim=dataset_info['num_features'],
            edge_attr_dim=dataset_info['edge_attr_dim'] if dataset_info['edge_attr'] else None,
            edge_attr=dataset_info['edge_attr'] if dataset_info['edge_attr'] else None,
            dropout=config['dropout_tea'],
            pooling_method=config['tea_pool_type'],
            useEmbedding=False,
            torchEmbedding=False,
            # addVirtualNode=True,
            # add_residual=False
            addVirtualNode=False,
            add_residual=True
        )
    else:
        raise ValueError("不支持的教师模型")
        
    # 构造学生模型
    if config['student_model'] == 'MLP':
        student_model = MLP(
            input_dim=dataset_info['num_features'],
            hidden_dim=config['stu_hidden_dim'],
            num_classes=dataset_info['num_classes'],
            num_layers=config['stu_layers'],
            dropout=config['dropout_stu'],
            pooling=config['stu_pool_type'],
            useEmbedding=False,
            add_residual=False
        )
    else:
        raise ValueError("不支持的学生模型")

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # 加载教师模型权重 (对应当前 fold)
    teacher_path = f"./output/{config['dataset_name']}/{config['teacher_model']}/seed{fold}/best_model.pth"
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"成功加载教师模型: {teacher_path}")
    else:
        print(f"警告: 找不到当前 fold 的教师模型权重 {teacher_path}")

    student_proj_head = ProjectionMLP(config['stu_hidden_dim'], config['proj_dim']).to(device)
    # teacher_proj_head = ProjectionMLP(config['tea_hidden_dim'], config['proj_dim']).to(device)

    optimizer = Adam(list(student_model.parameters()) + list(student_proj_head.parameters()), 
                     lr=config['lr'], weight_decay=config['weight_decay'])
    # optimizer = Adam(list(student_model.parameters()), 
    #                  lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    save_dir = os.path.join(base_output_dir, f"fold{fold}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(config['epochs']):
        train_metrics_list = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}', leave=False, file=sys.stderr)
        
        for batch in train_pbar:
            metrics = train_step(teacher_model, student_model, batch, optimizer, device, config, student_proj_head)
            train_metrics_list.append(metrics)
            
        avg_train_metrics = {k: float(np.mean([m[k] for m in train_metrics_list])) for k in train_metrics_list[0].keys()}
        
        val_metrics = evaluate_model(student_model, val_loader, device)
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1:03d} | Total Loss: {avg_train_metrics['total_loss']:.4f} | Task Loss: {avg_train_metrics['task_loss']:.4f} | "
                  f"KL: {avg_train_metrics['kl_loss']:.4f} | Core: {avg_train_metrics['loss_core_rkd']:.4f} | "
                  f"SubG Feat: {avg_train_metrics['loss_sub_feat']:.4f} | SubG Rel: {avg_train_metrics['loss_sub_rel']:.4f} | "
                  f"TopK: {avg_train_metrics['loss_topk']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            best_model_state = student_model.state_dict()
            torch.save(best_model_state, best_model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early Stopping at Epoch {epoch+1}")
            break
            
    if best_model_state:
        student_model.load_state_dict(best_model_state)
        
    test_metrics = evaluate_model(student_model, test_loader, device)
    teacher_test_metrics = evaluate_model(teacher_model, test_loader, device)
    print(f"Fold {fold} Finish. Test Acc: {test_metrics['accuracy']:.4f}")
    
    return test_metrics['accuracy']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--root', type=str, default='./TUdatasets', help='Root directory for datasets')
    
    # Teacher params (used to load the model)
    parser.add_argument('--teacher_model', type=str, default='GIN')
    parser.add_argument('--tea_layers', type=int, default=5)
    parser.add_argument('--tea_hidden_dim', type=int, default=128)
    parser.add_argument('--dropout_tea', type=float, default=0.2)
    parser.add_argument('--tea_pool_type', type=str, default='add')
    
    # Student params
    parser.add_argument('--student_model', type=str, default='MLP')
    parser.add_argument('--stu_layers', type=int, default=5)
    parser.add_argument('--stu_hidden_dim', type=int, default=128)
    parser.add_argument('--dropout_stu', type=float, default=0.2)
    parser.add_argument('--stu_pool_type', type=str, default='mean')
    
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--use_degree', action='store_true')
    
    # KD weights
    parser.add_argument('--task_weight', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--core_nce_weight', type=float, default=0.0)
    parser.add_argument('--subgraph_weight', type=float, default=0.0)
    parser.add_argument('--topk_weight', type=float, default=0.0)

    parser.add_argument('--kl_temperature', type=float, default=1.0)
    parser.add_argument('--sub_temperature', type=float, default=0.1)
    parser.add_argument('--rank_temperature', type=float, default=0.1)
    parser.add_argument('--core_temperature', type=float, default=2)

    parser.add_argument('--file', type=str, default="adaptArg", choices=['adaptArg','M3-RKD','AI','ALL','T'], help='Log file prefix to distinguish different methods')
    parser.add_argument('--device',type=int,default=0)

    args = parser.parse_args()
    
    # 转换为小写以实现大小写无关的比较
    dataset_name_lower = args.dataset.lower()
    
    if dataset_name_lower == 'mutagenicity':
        auto_n_clusters = 6
    elif dataset_name_lower in ['aids', 'frankenstein']:
        auto_n_clusters = 3  # 图较小，聚类数减少
    elif dataset_name_lower == 'collab':
        auto_n_clusters = 10
    elif dataset_name_lower in ['reddit-multi-5k', 'reddit-binary']:
        auto_n_clusters = 20
    else:
        auto_n_clusters = 5 # 默认兜底

    config = vars(args)
    config['dataset_name'] = args.dataset
    
    if dataset_name_lower in ['mutagenicity', 'aids', 'frankenstein', 'reddit-binary']:
        auto_temp = 1.0  # 二分类，不需要太软
    elif dataset_name_lower in ['collab']:
        auto_temp = 2.0  
    elif dataset_name_lower == 'reddit-multi-5k':
        auto_temp = 3.5  # 五分类，重度软化提取类间暗知
    elif dataset_name_lower == 'imdb-binary':
        auto_temp = 1.0  # IMDB-BINARY 的温度设置
    else:
        auto_temp = 2.0  # 默认兜底

    config['kl_temperature'] = auto_temp

# 根据数据集物理特性，自动分配 Top-K 的微观感受野
    if dataset_name_lower == 'mutagenicity':
        auto_top_k = 8
    elif dataset_name_lower in ['aids', 'enzymes','imdb-binary']:
        auto_top_k = 5
    elif dataset_name_lower == 'frankenstein':
        auto_top_k = 6
    elif dataset_name_lower == 'collab':
        auto_top_k = 15
    elif dataset_name_lower in ['reddit-multi-5k', 'reddit-binary']:
        auto_top_k = 25
    else:
        auto_top_k = 10 # 默认兜底

    config['topk_k'] = auto_top_k

    if dataset_name_lower in ['enzymes', 'mutagenicity', 'imdb-binary', 'reddit-binary', 'collab', 'reddit-multi-5k']:
        config['use_degree'] = True
    else:
        config['use_degree'] = False

    if args.file == "M3-RKD":
        log_dir = f'log_kd_TUdatasets/{args.dataset}/M3-RKD'
    elif args.file == "adaptArg":
        log_dir = f'log_kd_TUdatasets/{args.dataset}/adaptArg'
    
    elif args.file == "AI":
        log_dir = f'log_kd_TUdatasets/{args.dataset}/AI'
    elif args.file == "ALL":
        log_dir = f'log_kd_TUdatasets/{args.dataset}/M3-RKD-ALL'
    elif args.file == "T":
        log_dir = f'log_kd_TUdatasets/{args.dataset}/M3-RKD-T'
    else:
        log_dir = f'log_kd_TUdatasets/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = (
        f"{log_dir}/KD_{config['teacher_model']}_to_{config['student_model']}_{args.stu_pool_type}_{args.stu_hidden_dim}_{args.dropout_stu}/"
        f"KL{config['kl_weight']}_Core{config['core_nce_weight']}_Sub{config['subgraph_weight']}_TopK{config['topk_weight']}_"
        f'klT{config["kl_temperature"]}_subT{config["sub_temperature"]}_rankT{config["rank_temperature"]}_coreT{config["core_temperature"]}.txt'
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        print(f"Config: {config}")
        
        full_dataset = load_tu_dataset(
            name=config['dataset_name'],
            root=config['root'],
            batch_size=config['batch_size'],
            use_degree=config['use_degree'],
            pe_dim=0,
            add_spd=False,
            add_core=True,
            add_spectral=True,
            n_clusters=auto_n_clusters
        )
        
        edge_attr_dim = 0
        if hasattr(full_dataset[0], 'edge_attr') and full_dataset[0].edge_attr is not None:
            if full_dataset[0].edge_attr.dim() > 1:
                edge_attr_dim = full_dataset[0].edge_attr.shape[1]
            else:
                edge_attr_dim = 1
                
        dataset_info = {
                        'num_features': full_dataset.num_features, 
                        'num_classes': full_dataset.num_classes,
                        'edge_attr': hasattr(full_dataset[0], 'edge_attr') and full_dataset[0].edge_attr is not None,
                        'edge_attr_dim': edge_attr_dim,
                    }
        print(f"Num Features: {dataset_info['num_features']}")
        print(f"Num Classes: {dataset_info['num_classes']}")
        
        labels = [data.y.item() for data in full_dataset]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12345)

        base_output_dir = f"./output/{args.dataset}/KD_{config['teacher_model']}_{config['student_model']}_{args.stu_pool_type}_{args.stu_hidden_dim}_{args.dropout_stu}/KL{config['kl_weight']}_Core{config['core_nce_weight']}_Sub{config['subgraph_weight']}_TopK{config['topk_weight']}"
        all_accs = []
        
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            train_val_labels = [labels[i] for i in train_val_idx]

            skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold)
            
            split_iter = skf_inner.split(np.zeros(len(train_val_idx)), train_val_labels)
            train_idx_rel, val_idx_rel = next(split_iter)
            
            train_idx = train_val_idx[train_idx_rel]
            val_idx = train_val_idx[val_idx_rel]
            
            # 使用Subset来保持原始图对象
            from torch.utils.data import Subset
            train_subset = Subset(full_dataset, train_idx.tolist())
            val_subset = Subset(full_dataset, val_idx.tolist())
            test_subset = Subset(full_dataset, test_idx.tolist())
            
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)
            
            acc = train_and_evaluate_kd(config, device, base_output_dir, fold, train_loader, val_loader, test_loader, dataset_info)
            all_accs.append(acc)
            
        print("\n" + "#"*60)
        print(f"Final KD Results on {args.dataset}:")
        print(f"Mean Acc: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
        print(f"Detailed: {all_accs}")
        print("#"*60)
    
    sys.stdout = original_stdout
    print(f"Done. Log saved to {log_file}")

if __name__ == "__main__":
    main()
