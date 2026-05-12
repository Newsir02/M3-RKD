import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm
import sys

from ogb.graphproppred import Evaluator
from data_util.load_ogbn_data import load_ogbg_dataset
from model.baseGNN import GIN, GIN_Drop,GCN,SAGE
from model.MLP import StudentMLP_with_PE,MLP
from kd_loss import nce_criterion, random_walk_distillation_loss, kl_divergence_loss,graph_topology_attention_loss, compute_core_view_graph_emb, rkd_distance_loss, subgraph_distillation_loss, topk_ranking_distillation_loss
from torch_geometric.utils import subgraph
from torch_scatter import scatter_max
import argparse
import copy

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


def evaluate_model(model, data_loader, device, num_tasks, desc='Evaluating', dataset_name: str = 'ogbg-molpcba'):
    model.eval()
    evaluator = Evaluator(name=dataset_name)
    y_true_list = []
    y_pred_list = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            batch = batch.to(device)
            if hasattr(batch, 'x'):
                if dataset_name.startswith('ogbg-mol'):
                    batch.x = batch.x.long()
                else:
                    batch.x = batch.x.float()
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                if dataset_name.startswith('ogbg-mol'):
                    batch.edge_attr = batch.edge_attr.long()
                else:
                    batch.edge_attr = batch.edge_attr.float()

            logits, _, _ = model(batch, True)

            if dataset_name == 'ogbg-molpcba':
                y = batch.y.float()
                mask = ~torch.isnan(y)
                y_clean = torch.where(mask, y, torch.zeros_like(y))
                loss_raw = criterion(logits, y_clean)
                loss = masked_task_mean(loss_raw, mask)
            else:
                y = batch.y.view(-1, 1).to(logits.device).float()
                mask = ~torch.isnan(y)
                if mask.sum() == 0:
                    continue
                loss = criterion(logits[mask], y[mask]).sum() / mask.sum()

            total_loss += float(loss.item())

            if dataset_name == 'ogbg-molpcba':
                y_pred = torch.sigmoid(logits).detach().cpu()
                y_true = y.detach().cpu()
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
            elif dataset_name == 'ogbg-molhiv':
                y_pred = logits.view(-1,1).cpu()
                y_true = batch.y.view(-1,1).cpu()
                y_true_list.append(y_true) 
                y_pred_list.append(y_pred)
            elif dataset_name in  ['ogbg-molbace', 'ogbg-molbbbp']:
                y_true = batch.y.view(logits.shape).cpu()
                y_pred = logits.cpu()
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

    y_true = torch.cat(y_true_list, dim=0).cpu()
    y_pred = torch.cat(y_pred_list, dim=0).cpu()
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    avg_loss = total_loss / len(data_loader)
    metric = result['ap'] if 'ap' in result else result.get('rocauc', None)
    out = {'metric': metric, 'Eva loss': avg_loss}
    if 'ap' in result:
        out['ap'] = result['ap']
    if 'rocauc' in result:
        out['rocauc'] = result['rocauc']
    return out


def masked_task_mean(loss_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lr = loss_raw * mask.float()
    per_task_sum = lr.sum(dim=0)
    per_task_cnt = mask.sum(dim=0).float()
    valid = per_task_cnt > 0
    per_task_mean = torch.zeros_like(per_task_sum)
    per_task_mean[valid] = per_task_sum[valid] / per_task_cnt[valid]
    if valid.any():
        return per_task_mean[valid].mean()
    else:
        return per_task_mean.mean()


def train_step(teacher_model, student_model, student_proj_head, teacher_proj_head, batch, optimizer, device,
               dataset_name='ogbg-molpcba', config=None):
    teacher_model.eval()
    student_model.train()
    optimizer.zero_grad()

    batch = batch.to(device)
    if dataset_name.startswith('ogbg-mol'):
    # OGB 数据集必须转为 Long 才能进 Encoder
    # 这一步其实 PyG 的 loader 可能已经做好了，但在 GPU 上确保一下没问题
        if batch.x is not None: batch.x = batch.x.long()
        if batch.edge_attr is not None: batch.edge_attr = batch.edge_attr.long()
    else:
        if batch.x is not None: batch.x = batch.x.float()
        if batch.edge_attr is not None: batch.edge_attr = batch.edge_attr.float()
    
    if hasattr(batch, 'core_atom_index'):
        core_global_indices = batch.core_atom_index.view(-1)
    else:
        core_global_indices = None

    if config['kl_weight'] > 0 or config['path_weight'] > 0 or config['topo_weight'] > 0 or config['core_nce_weight'] > 0 or config['subgraph_weight'] > 0 or config['topk_weight'] > 0:
        with torch.no_grad():
            t_logits, t_node_emb, t_graph_emb = teacher_model(batch, True)
    s_logits, s_node_emb, s_graph_emb,s_attns = student_model(batch, False, return_attn=True)

    # 任务损失（按数据集类型）
    if dataset_name == 'ogbg-molpcba':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        y = batch.y.float()
        mask = ~torch.isnan(y)
        y_clean = torch.where(mask, y, torch.zeros_like(y))
        task_loss = masked_task_mean(criterion(s_logits, y_clean), mask)
    elif dataset_name == 'ogbg-molhiv':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        y = batch.y.view(-1, 1).float().to(s_logits.device)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            task_loss = torch.tensor(0.0, device=s_logits.device)
        else:
            task_loss = criterion(s_logits[mask], y[mask])

    elif dataset_name in  ['ogbg-molbace', 'ogbg-molbbbp']:
            y = batch.y.view(s_logits.shape).float()
            task_loss = F.binary_cross_entropy_with_logits(s_logits, y)

    total_kd_loss = 0.0
    kl_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    topo_loss = torch.tensor(0.0, device=device)
    loss_core_rkd = torch.tensor(0.0, device=device)
    loss_sub_feat = torch.tensor(0.0, device=device)
    loss_sub_rel = torch.tensor(0.0, device=device)
    loss_topk = torch.tensor(0.0, device=device)
    # KL 蒸馏
    if config['kl_weight'] > 0:
        kl_loss = kl_divergence_loss(s_logits, t_logits, temperature=config['kl_temperature']) if config['kl_weight'] > 0 else torch.tensor(0.0, device=device)
        total_kd_loss += config['kl_weight'] * kl_loss

    
    if config['path_weight'] > 0 or config['topo_weight'] > 0 or config['core_nce_weight'] > 0 or config['subgraph_weight'] > 0 or config['topk_weight'] > 0:
        # A. 一次性计算整个 Batch 的投影 (利用 GPU 并行优势)
        # 这里的 t_node_emb 是 [Batch_Total_Nodes, Hidden_Dim]
        t_proj = teacher_proj_head(t_node_emb)
        s_proj = student_proj_head(s_node_emb)

        #  计算 Topo Loss (基于注意力图)
        if config['topo_weight'] > 0:
            if hasattr(batch, 'spatial_pe_idx'):
                topo_loss = graph_topology_attention_loss(
                    s_attns, 
                    batch, 
                    sigma=2.0
                )
        # C. 计算 Path Loss (批量随机游走)
        if config['path_weight'] > 0:
            # 策略：为了避免对所有节点计算（太慢），我们从整个 Batch 中随机采样节点
            # 比如采样 512 个节点作为 Center Node 进行游走
            # 这种方式比逐图循环快得多，且无需 subgraph
            
            total_nodes = t_proj.shape[0]
            num_samples = min(512, total_nodes) # 你可以根据显存调整这个数字，越大越准
            
            # 随机采样节点索引
            # 确保在 device 上生成
            node_indices = torch.randperm(total_nodes, device=device)[:num_samples]
            
            # 直接传入 batch.edge_index
            # 因为 Batch 图是互不连通的，游走绝对不会跨图，逻辑完全正确
            path_loss = random_walk_distillation_loss(
                t_graph_emb, 
                s_graph_emb, 
                batch.edge_index,  # 直接使用全图 edge_index
                n=10, 
                k=5,
                node_indices=node_indices, # 只对采样的节点计算 Loss
                similarity_func='guss_euclidean', 
                loss_type='relaxed_mse',
                temperature=2,
                threshold=config['threshold']
            )
        
        if config['core_nce_weight'] > 0:
            teacher_core_emb = compute_core_view_graph_emb(t_node_emb, batch.batch, core_global_indices,t=config['core_temperature'])
            student_core_emb = compute_core_view_graph_emb(s_node_emb, batch.batch, core_global_indices,t=config['core_temperature'])
            loss_core_rkd = rkd_distance_loss(teacher_core_emb.detach(), student_core_emb, delta=1.0)
            #loss_core_rkd = rkd_distance_loss(t_graph_emb.detach(), s_graph_emb, delta=1.0)
        
        if config['subgraph_weight'] > 0 and hasattr(batch, 'fragment_index'):
  
            sparse_idx = batch.fragment_index
            _, dense_idx = torch.unique(sparse_idx, return_inverse=True)
            global_frag_index = dense_idx
            # --- 计算 Loss ---
            loss_sub_feat, loss_sub_rel = subgraph_distillation_loss(
                t_node_emb, 
                s_node_emb, 
                global_frag_index, 
                batch.batch, 
                loss_type='kl',
                temperature=config['sub_temperature']
            )
        if config.get('topk_weight', 0) > 0:
        # 使用投影头后的特征计算 (推荐)，或者直接用 node_emb
        # 推荐使用 proj 后的特征，因为 node_emb 要负责回归任务，空间可能受限
            # t_feat = teacher_proj_head(t_node_emb)
            # s_feat = student_proj_head(s_node_emb)
        
            # loss_topk = topk_ranking_distillation_loss(
            #     t_feat, 
            #     s_feat, 
            #     batch.batch, 
            #     k=8,  # ZINC 平均度数约为 2-3，但考虑到 2-hop，K=8 比较合适
            #     temperature=config['temperature']
            # )
            loss_topk = topk_ranking_distillation_loss(
                t_node_emb,
                s_node_emb, 
                batch.batch, 
                k=config['k'],  # ZINC 平均度数约为 2-3，但考虑到 2-hop，K=8 比较合适
                temperature=config['rank_temperature']
            )

    total_kd_loss += config['path_weight'] * path_loss + config['topo_weight'] * topo_loss + config['core_nce_weight'] * loss_core_rkd + config['subgraph_weight'] * (loss_sub_feat + loss_sub_rel) + config['topk_weight']* loss_topk

    total_loss = config['task_weight'] * task_loss + total_kd_loss
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'kd_loss': total_kd_loss.item(),
        'kl_loss': kl_loss.item(),
        'topo_loss': topo_loss.item(),
        'path_loss': path_loss.item(),
        'loss_core_rkd': loss_core_rkd.item(),
        'loss_sub_feat': loss_sub_feat.item(),
        'loss_sub_rel': loss_sub_rel.item(),
        'subgraph_loss': (loss_sub_feat + loss_sub_rel).item(),
        'loss_topk': loss_topk.item()
    }


def train_and_evaluate_kd(config, device, base_output_dir, run_idx=0, train_loader=None, val_loader=None, test_loader=None, dataset_info=None):
    seed = config['seed']
    print(f"\n{'='*40}")
    print(f"开始第 {run_idx+1}/{len(config['seeds'])} 次运行 | Seed: {seed}")
    print(f"{'='*40}")

    print('加载数据集完成')
    
    print('数据集信息:')
    print(f"  - 图数量: {dataset_info['num_graphs']}")
    print(f"  - 任务数: {dataset_info['num_tasks']}")
    print(f"  - 节点特征维度: {dataset_info['num_features']}")
    print(f"  - 平均节点数: {dataset_info['avg_nodes']:.1f}")
    #print(f"  - 平均边数: {dataset_info['avg_edges']:.1f}")

    # 构建教师模型
    def build_model(kind, hidden_dim):
        if kind == 'GIN':
            return GIN(
                num_layers=config['tea_layers'],
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_tasks'],
                input_dim=dataset_info['num_features'],
                dropout=config['dropout_tea'],
                pooling_method=config['tea_pool_type'],
                edge_attr=dataset_info['edge_attr'],
                edge_attr_dim=dataset_info['edge_attr_dim'],
                useEmbedding=True,
                addVirtualNode=True,
            )
        elif kind == 'GIN_drop':
            return GIN_Drop(
                num_layers=config['tea_layers'],
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_tasks'],
                input_dim=dataset_info['num_features'],
                dropout=config['dropout_tea'],
                pooling_method=config['tea_pool_type'],
                edge_attr=dataset_info['edge_attr'],
                edge_attr_dim=dataset_info['edge_attr_dim'],
                useEmbedding=True,
                addVirtualNode=True,
            )
        elif kind == 'GCN':
            return GCN(
                num_layers=config['tea_layers'],
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_tasks'],
                input_dim=dataset_info['num_features'],
                dropout=config['dropout_tea'],
                pooling_method=config['tea_pool_type'],
                edge_attr=dataset_info['edge_attr'],
                edge_attr_dim=dataset_info['edge_attr_dim'],
                useEmbedding=True,
                addVirtualNode=True,
            )
        
        elif kind == 'SAGE':
            return SAGE(
                num_layers=config['tea_layers'],
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_tasks'],
                input_dim=dataset_info['num_features'],
                dropout=config['dropout_tea'],
                pooling_method=config['tea_pool_type'],
                edge_attr=dataset_info['edge_attr'],
                edge_attr_dim=dataset_info['edge_attr_dim'],
                useEmbedding=True,
                addVirtualNode=True,
                add_residual=False
            )
        
        elif kind == 'MLP_with_PE':
            return StudentMLP_with_PE(
                input_dim=dataset_info['num_features'],
                pe_dim=config['pe_dim'],
                hidden_dim=hidden_dim,
                output_dim=dataset_info['num_tasks'],
                num_layers=config['stu_layers'],
                dropout=config['dropout_stu'],
                pool_type=config['stu_pool_type'],
                use_embedding=True,
                add_pe=config['add_pe'],
            )
        elif kind == 'MLP':
            return MLP(
                    input_dim=dataset_info["num_features"],
                    hidden_dim=hidden_dim,
                    num_classes=dataset_info["num_tasks"],
                    num_layers=config['stu_layers'],
                    dropout=config['dropout_stu'],
                    pooling=config['stu_pool_type'],
                )
        else:
            raise ValueError(f'不支持的模型类型: {kind}')

    teacher_model = build_model(config['teacher_model'], config['tea_hidden_dim']).to(device)
    student_model = build_model(config['student_model'], config['stu_hidden_dim']).to(device)

    # 可选：加载教师权重
    if 'teacher_path' in config and config['teacher_path']:
        if os.path.exists(config['teacher_path']):
            teacher_model.load_state_dict(torch.load(config['teacher_path'], map_location=device))
            print(f"成功加载教师模型: {config['teacher_path']}")
        else:
            print(f"警告: 教师模型文件不存在: {config['teacher_path']}")

    print(f"教师参数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"学生参数量: {sum(p.numel() for p in student_model.parameters()):,}")

    # 投影头（用于 NCE/Path）
    student_proj_head = ProjectionMLP(config['stu_hidden_dim'], config['proj_dim']).to(device)
    teacher_proj_head = ProjectionMLP(config['tea_hidden_dim'], config['proj_dim']).to(device)

    # optimizer = Adam(list(student_model.parameters()) + list(student_proj_head.parameters()) +  list(teacher_proj_head.parameters()) , 
    #                  lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer = Adam(list(student_model.parameters()) + list(student_proj_head.parameters()) , 
                     lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config['patience'])

    history = {
        'epoch': [],
        'train_loss': [],
        'train_task_loss': [],
        'train_kd_loss': [],
        'train_kl_loss': [],
        'train_path_loss': [],
        'train_topo_loss': [],
        'train_topk_loss': [],
        'val_metric': [],
        'val_loss': [],
        'lr': []
    }
    best_val_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 创建 Seed 专属目录
    save_dir = os.path.join(base_output_dir, f"run_seed{config['seed']}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    history_path = os.path.join(save_dir, 'training_history.csv')

    print("\n开始KD训练...")
    start_time = time.time()
    epochs = config['epochs']
    for epoch in range(epochs):
        train_metrics_list = []
        # 使用 sys.stderr 确保进度条在重定向 stdout 时仍然可见
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]', leave=False, file=sys.stderr)

        for batch in train_pbar:

            metrics = train_step(
                teacher_model, student_model, student_proj_head, teacher_proj_head,
                batch, optimizer, device,
                dataset_name=config['dataset_name'],config=config)
            
            train_metrics_list.append(metrics)
            train_pbar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                'Task': f"{metrics['task_loss']:.4f}",
                'KD': f"{metrics['kd_loss']:.4f}",
                'KL': f"{metrics['kl_loss']:.4f}",
                'PATH': f"{metrics['path_loss']:.4f}",
                'TOPO': f"{metrics['topo_loss']:.4f}",
                'CoreNCE': f"{metrics['loss_core_rkd']:.4f}",
                'SubG': f"{metrics['subgraph_loss']:.4f}",
                'TopK': f"{metrics['loss_topk']:.4f}"
            })

        avg_train_metrics = {k: float(np.mean([m[k] for m in train_metrics_list])) for k in train_metrics_list[0].keys()}

        val_metrics = evaluate_model(student_model, val_loader, device, dataset_info['num_tasks'], desc='Validating', dataset_name=config['dataset_name'])
        scheduler.step(val_metrics['metric'])
        current_lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_metrics['total_loss'])
        history['train_task_loss'].append(avg_train_metrics['task_loss'])
        history['train_kd_loss'].append(avg_train_metrics['kd_loss'])
        history['train_kl_loss'].append(avg_train_metrics['kl_loss'])
        history['train_path_loss'].append(avg_train_metrics['path_loss'])
        history['train_topo_loss'].append(avg_train_metrics['topo_loss'])
        history['train_subgraph_loss'] = history.get('train_subgraph_loss', []) + [avg_train_metrics['subgraph_loss']]
        history['train_topk_loss'].append(avg_train_metrics['loss_topk'])
        history['val_metric'].append(val_metrics['metric'])
        history['val_loss'].append(val_metrics['Eva loss'])
        history['lr'].append(current_lr)

        is_best = val_metrics['metric'] > best_val_metric

        if (epoch + 1) % 10 == 0 or is_best:
            print(
                f"Run {run_idx+1} Epoch {epoch+1:3d} | "
                f"Total: {avg_train_metrics['total_loss']:.4f} | "
                f"Task: {avg_train_metrics['task_loss']:.4f} | "
                f"KL: {avg_train_metrics['kl_loss']:.4f} | "
                f"Topo: {avg_train_metrics['topo_loss']:.4f} | "
                f"Path: {avg_train_metrics['path_loss']:.4f} | "
                f"CoreNCE: {avg_train_metrics['loss_core_rkd']:.4f} | "
                f"SubG: {avg_train_metrics['subgraph_loss']:.4f} | "
                f"TopK: {avg_train_metrics['loss_topk']:.4f} | "
                f"Val Metric: {val_metrics['metric']:.4f} | "
                #f"Decay: {decay_factor:.2f}"
            )

        if is_best:
            best_val_metric = val_metrics['metric']
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  >>> New Best Val Acc: {best_val_metric:.4f} (Saved)")
            best_model_state = student_model.state_dict()
            
        else:
            patience_counter += 1
        if patience_counter >= config['patience']:
            print(f"\n早停触发！在第 {epoch+1} 轮停止训练 (Best Val: {best_val_metric:.4f})")
            break

    training_time = time.time() - start_time

    df_history = pd.DataFrame(history)
    df_history.to_csv(history_path, index=False)

    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
        best_model_state = student_model.state_dict()
        torch.save(best_model_state, best_model_path)

    test_metrics = evaluate_model(student_model, test_loader, device, dataset_info['num_tasks'], desc='Testing', dataset_name=config['dataset_name'])

    # 评估教师模型
    teacher_test_metrics = evaluate_model(teacher_model, test_loader, device, dataset_info['num_tasks'], desc='Testing Teacher', dataset_name=config['dataset_name'])

    print(f"训练完成! 最佳验证: {best_val_metric:.4f} (epoch {best_epoch}), 用时: {training_time:.2f}s")
    print(f"测试 Metric: {test_metrics['metric']:.4f}")
    print(f"训练历史保存路径: {history_path}")
    print(f"最佳模型权重保存路径: {best_model_path}")

    return test_metrics, best_val_metric, best_model_state, best_epoch


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ogbg-molhiv', help='Dataset name')
    
    # Teacher params (used to load the model)
    parser.add_argument('--teacher_model', type=str, default='GIN')
    parser.add_argument('--tea_layers', type=int, default=5)
    parser.add_argument('--tea_hidden_dim', type=int, default=256)
    parser.add_argument('--dropout_tea', type=float, default=0.5)
    parser.add_argument('--tea_pool_type', type=str, default='add')
    parser.add_argument('--teacher_path', type=str, default='output/GIN/ogbg-molhiv_hidden256_layers5_dropout0.5_pooladd/run4/ogbg-molhiv_best_model_run4.pth')
    
    # Student params
    parser.add_argument('--student_model', type=str, default='MLP')
    parser.add_argument('--stu_layers', type=int, default=5)
    parser.add_argument('--stu_hidden_dim', type=int, default=256)
    parser.add_argument('--dropout_stu', type=float, default=0.5)
    parser.add_argument('--stu_pool_type', type=str, default='mean')
    
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    
    parser.add_argument('--pe_dim', type=int, default=8)
    parser.add_argument('--add_pe', action='store_true', default=False)
    parser.add_argument('--add_spd', action='store_true', default=True)
    parser.add_argument('--add_core', action='store_true', default=False)
    parser.add_argument('--add_spectral', action='store_true', default=False)
    
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--n_clusters', type=int, default=5)
    
    # KD weights
    parser.add_argument('--task_weight', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--core_nce_weight', type=float, default=0.7)
    parser.add_argument('--subgraph_weight', type=float, default=0.0)
    parser.add_argument('--topk_weight', type=float, default=10.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    
    parser.add_argument('--topo_weight', type=float, default=0.0)
    parser.add_argument('--path_weight', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--min_weight', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.05)
    
    parser.add_argument('--kl_temperature', type=float, default=1.0)
    parser.add_argument('--sub_temperature', type=float, default=0.1)
    parser.add_argument('--rank_temperature', type=float, default=0.1)
    parser.add_argument('--core_temperature', type=float, default=2)
    parser.add_argument('--k', type=int, default=8, help='Top-K for ranking distillation')

    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--iteration', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    config = vars(args)
    config['seeds'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 优先选择 cuda
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # 清理显存，防止之前的进程残留
    
    
    train_loader_once, val_loader_once, test_loader_once, dataset_info_once = load_ogbg_dataset(
        name=config['dataset_name'],
        batch_size=config['batch_size'],
        pe_dim=config['pe_dim'],
        add_spd=config['add_spd'],
        add_core=config['add_core'],
        add_spectral=config['add_spectral'],
        subset_ratio=config['subset_ratio'],
        n_clusters=config['n_clusters'],
        force_reload=False
    )
    # train_loader_once, val_loader_once, test_loader_once, dataset_info_once = load_ogbg_dataset(
    #     name=config['dataset_name'], 
    #     batch_size=config['batch_size'],
    #     pe_dim=config['pe_dim'], 
    #     add_spd=config['add_spd'], 
    #     add_core=True,
    #     add_spectral=True,
    #     subset_ratio=config['subset_ratio'],
    #     n_clusters=config['n_clusters'],
    #     force_reload=True
    # )

    if True:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_dir = f'MLPlog-{config["dataset_name"]}'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = (
            f'{log_dir}/training_log_Experiment_M3-RKD-k_{config["dataset_name"]}_'
            f'{config["teacher_model"]}_{config["tea_hidden_dim"]}_{config["tea_layers"]}_{config["dropout_tea"]}iteration{config["iteration"]}_{config["student_model"]}_{config["stu_layers"]}_{config["dropout_stu"]}_addpe{config["add_pe"]}/'
            f'pool-{config["k"]}_{config["stu_pool_type"]}_kl{config["kl_weight"]}_core{config["core_nce_weight"]}_sub{config.get("subgraph_weight", 0.0)}_topk{config.get("topk_weight", 0.0)}_'
            f'{config["kl_temperature"]}_{config["sub_temperature"]}_{config["rank_temperature"]}_{config["core_temperature"]}'
            f'klT{config["kl_temperature"]}_subT{config["sub_temperature"]}_rankT{config["rank_temperature"]}_coreT{config["core_temperature"]}.txt'
        )
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        output_dir = config.get('output_dir', f'./Expriment_output-K/MLPDistillationOGB-{config["dataset_name"]}')
        son_dir = f"{config['stu_pool_type']}_{config['dataset_name']}_{config['teacher_model']}_{config['tea_hidden_dim']}_{config['tea_layers']}_iteration{config['iteration']}_{config['student_model']}_{config['stu_hidden_dim']}{config['stu_layers']}_{config['dropout_stu']}"
        save_dir = f"KL{config['kl_weight']}_Core{config['core_nce_weight']}_Sub{config['subgraph_weight']}_TopK{config['topk_weight']}"
        base_output_dir = os.path.join(output_dir, son_dir, save_dir)
        os.makedirs(base_output_dir, exist_ok=True)

        original_stdout = sys.stdout  
        
        # 使用 with 打开文件，并将 stdout 重定向到文件
        # 这样所有 print() 输出都会实时写入文件
        with open(log_file_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            print(f"Configuration: {config}")
            print(f"Device: {device}")
            print(f"Log file: {log_file_path}")
            print(f"Output directory: {base_output_dir}")

            # 1. 初始化结果列表
            all_test_metrics = []
            
            # 2. 开始多种子循环
            for run_idx, seed in enumerate(config['seeds']):
                # 设置种子
                set_seed(seed)
                config['seed'] = seed
                
                # 运行训练
                test_metric, _, _, _ = train_and_evaluate_kd(
                    config, device, base_output_dir, run_idx,
                    train_loader=train_loader_once,
                    val_loader=val_loader_once,
                    test_loader=test_loader_once,
                    dataset_info=dataset_info_once
                )
                
                # 收集结果
                all_test_metrics.append(test_metric['metric'])

            # 3. 计算均值和标准差
            mean_score = np.mean(all_test_metrics)
            std_score = np.std(all_test_metrics)
            
            # 4. 格式化输出信息
            print(f"\n{'#'*60}")
            print(f"最终结果汇总 ({len(config['seeds'])} runs):")
            print(f"Mean Test Metric: {mean_score:.4f} ± {std_score:.4f}")
            print(f"详细 Metrics: {all_test_metrics}")
            print(f"{'#'*60}")
            
            # 5. 追加保存到 CSV (final_summary_results.csv)
            son_csv = f"{config['dataset_name']}_Base_{config['stu_pool_type']}_final_summary_results.csv"
            results_csv_path = os.path.join(output_dir, son_csv)
                                            
            summary_data = {
                'dataset': config['dataset_name'],
                'teacher': config['teacher_model'],
                'student': f"{config['student_model']}+PE_{config['add_pe']}+{config['stu_layers']}+{config['stu_hidden_dim']}+{config['dropout_stu']}",
                'seeds_count': len(config['seeds']),
                'kl_w': config['kl_weight'],
                'path_w': config['path_weight'],
                'topo_w': config['topo_weight'],
                'core_nce_w': config['core_nce_weight'],
                'subgraph_w': config.get('subgraph_weight', 0.0),
                'topk_w': config.get('topk_weight', 0.0),
                'mean_metric': mean_score,
                'std_metric': std_score,
                'all_metrics': str(all_test_metrics)
            }
            
            df_result = pd.DataFrame([summary_data])
            if os.path.exists(results_csv_path):
                df_result.to_csv(results_csv_path, mode='a', header=False, index=False)
            else:
                df_result.to_csv(results_csv_path, mode='w', header=True, index=False)
                    
            print(f"汇总结果已保存至: {results_csv_path}")

            # 恢复 stdout
            sys.stdout = original_stdout
            print(f"Training finished. Log saved to {log_file_path}")


if __name__ == '__main__':
    main()
