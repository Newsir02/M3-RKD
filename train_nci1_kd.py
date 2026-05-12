import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import os
import time
from tqdm import tqdm
import sys
import random

# 引入模型
from model.baseGNN import GIN,GCN,SAGE
from model.MLP import MLP
from data_util.load_nci1_dataset import load_nci1_dataset
# from data_util.load_tu_datasets import load_tu_dataset

# 引入蒸馏损失
from kd_loss import (
    nce_criterion, 
    kl_divergence_loss,
    compute_core_view_graph_emb, 
    rkd_distance_loss, 
    subgraph_distillation_loss, 
    topk_ranking_distillation_loss
)


class ProjectionMLP(nn.Module):
    """投影头，用于 NCE 等对比学习损失"""
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
    """评估模型性能（准确率、精确率、召回率、F1）"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # 兼容不同模型的输入
            if isinstance(model, (GIN, MLP,GCN,SAGE)):
                logits= model(batch)
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
                
            loss = criterion(logits, batch.y)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            total_loss += loss.item()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    avg_loss = total_loss / len(data_loader)
    
    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'loss': avg_loss
    }


def train_step(teacher_model, student_model, student_proj_head, teacher_proj_head, 
               batch, optimizer, device, config):
    """单步训练（包含知识蒸馏）"""
    teacher_model.eval()
    student_model.train()
    optimizer.zero_grad()

    batch = batch.to(device)
    
    # 提取核心节点索引（如果存在）
    if hasattr(batch, 'core_atom_index'):
        core_global_indices = batch.core_atom_index.view(-1)
    else:
        core_global_indices = None

    # 教师模型前向传播
    if config['kl_weight'] > 0 or config['nce_weight'] > 0 or \
       config['core_nce_weight'] > 0 or config['subgraph_weight'] > 0 or config['topk_weight'] > 0:
        with torch.no_grad():
            t_logits, t_node_emb, t_graph_emb = teacher_model(batch, True)
    
    # 学生模型前向传播
    s_logits, s_node_emb, s_graph_emb = student_model(batch,True, return_attn=False)

    # 任务损失（分类）
    criterion = nn.CrossEntropyLoss()
    task_loss = criterion(s_logits, batch.y)

    total_kd_loss = 0.0
    kl_loss = torch.tensor(0.0, device=device)
    nce_loss = torch.tensor(0.0, device=device)
    loss_core_rkd = torch.tensor(0.0, device=device)
    loss_sub_feat = torch.tensor(0.0, device=device)
    loss_sub_rel = torch.tensor(0.0, device=device)
    loss_topk = torch.tensor(0.0, device=device)

    # KL 蒸馏
    if config['kl_weight'] > 0:
        kl_loss = kl_divergence_loss(s_logits, t_logits, temperature=1.0)
        total_kd_loss += config['kl_weight'] * kl_loss

    # 节点级蒸馏（NCE/Core/Subgraph/TopK）
    if config['nce_weight'] > 0 or config['core_nce_weight'] > 0 or \
       config['subgraph_weight'] > 0 or config['topk_weight'] > 0:
        
        # 投影节点嵌入
        t_proj = teacher_proj_head(t_node_emb)
        s_proj = student_proj_head(s_node_emb)

        # NCE Loss（全局对比）
        if config['nce_weight'] > 0:
            nce_loss = nce_criterion(s_proj, t_proj, nce_T=0.075, max_samples=4096)

        # Core RKD Loss（核心子图）
        if config['core_nce_weight'] > 0 and core_global_indices is not None:
            teacher_core_emb = compute_core_view_graph_emb(
                t_node_emb, batch.batch, core_global_indices
            )
            student_core_emb = compute_core_view_graph_emb(
                s_node_emb, batch.batch, core_global_indices
            )
            loss_core_rkd = rkd_distance_loss(
                teacher_core_emb.detach(), student_core_emb, delta=1.0
            )

        # Subgraph Loss（片段）
        if config['subgraph_weight'] > 0 and hasattr(batch, 'fragment_index'):
            sparse_idx = batch.fragment_index
            _, dense_idx = torch.unique(sparse_idx, return_inverse=True)
            global_frag_index = dense_idx
            
            loss_sub_feat, loss_sub_rel = subgraph_distillation_loss(
                t_node_emb, s_node_emb, global_frag_index, batch.batch,loss_type='kl',temperature=0.1
            )

        # TopK Ranking Loss
        if config['topk_weight'] > 0:
            loss_topk = topk_ranking_distillation_loss(
                t_node_emb, s_node_emb, batch.batch,
                k=config.get('topk_k', 8),
                temperature=config.get('temperature', 0.1)
            )

    # 汇总 KD 损失
    total_kd_loss += (
        config['nce_weight'] * nce_loss + 
        config['core_nce_weight'] * loss_core_rkd + 
        config['subgraph_weight'] * (loss_sub_feat + loss_sub_rel) + 
        config['topk_weight'] * loss_topk
    )

    # 总损失
    total_loss = config['task_weight'] * task_loss + total_kd_loss
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'kd_loss': total_kd_loss.item(),
        'kl_loss': kl_loss.item(),
        'nce_loss': nce_loss.item(),
        'loss_core_rkd': loss_core_rkd.item(),
        'loss_sub_feat': loss_sub_feat.item(),
        'loss_sub_rel': loss_sub_rel.item(),
        'subgraph_loss': (loss_sub_feat + loss_sub_rel).item(),
        'loss_topk': loss_topk.item()
    }


def train_and_evaluate_kd(config, device, base_output_dir, run_idx=0, 
                          train_loader=None, val_loader=None, test_loader=None, 
                          dataset_info=None):
    """完整的训练和评估流程"""
    seed = config['seed']
    print(f"\n{'='*40}")
    print(f"开始第 {run_idx+1}/{len(config['seeds'])} 次运行 | Seed: {seed}")
    print(f"{'='*40}")

    print('数据集信息:')
    print(f"  - 图数量: {dataset_info.get('num_graphs', 'N/A')}")
    print(f"  - 类别数: {dataset_info['num_classes']}")
    print(f"  - 节点特征维度: {dataset_info['num_features']}")

    # 构建模型
    def build_model(kind, hidden_dim, num_layers):
        if kind == 'GIN':
            return GIN(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_classes'],
                input_dim=dataset_info['num_features'],
                dropout=config.get('dropout_tea', 0.5),
                pooling_method=config.get('tea_pool_type', 'add'),
                useEmbedding=False,
                torchEmbedding=False,
                addVirtualNode=False,
                add_residual=True
            )
        elif kind == 'GCN':
            return GCN(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_classes'],
                input_dim=dataset_info['num_features'],
                dropout=config.get('dropout_tea', 0.5),
                pooling_method=config.get('tea_pool_type', 'add'),
                useEmbedding=False,
                torchEmbedding=False,
                addVirtualNode=False,
                add_residual=True
            )
        elif kind == 'SAGE':
            return SAGE(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_classes'],
                input_dim=dataset_info['num_features'],
                dropout=config.get('dropout_tea', 0.5),
                pooling_method=config.get('tea_pool_type', 'add'),
                useEmbedding=False,
                torchEmbedding=False,
                addVirtualNode=False,
                add_residual=True
            )
            
        elif kind == 'MLP':
            return MLP(
                input_dim=dataset_info['num_features'],
                hidden_dim=hidden_dim,
                num_classes=dataset_info['num_classes'],
                num_layers=num_layers,
                dropout=config.get('dropout_stu', 0.5),
                pooling=config.get('stu_pool_type', 'add'),
                useEmbedding=False,
                add_residual=(num_layers > 3)
            )
        else:
            raise ValueError(f'不支持的模型类型: {kind}')

    teacher_model = build_model(
        config['teacher_model'], 
        config['tea_hidden_dim'], 
        config['tea_layers']
    ).to(device)
    
    student_model = build_model(
        config['student_model'], 
        config['stu_hidden_dim'], 
        config['stu_layers']
    ).to(device)

    # 加载教师权重
    if 'teacher_path' in config and config['teacher_path']:
        if os.path.exists(config['teacher_path']):
            teacher_model.load_state_dict(torch.load(config['teacher_path'], map_location=device))
            print(f"成功加载教师模型: {config['teacher_path']}")
        else:
            print(f"警告: 教师模型文件不存在: {config['teacher_path']}")

    print(f"教师参数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"学生参数量: {sum(p.numel() for p in student_model.parameters()):,}")

    # 投影头
    student_proj_head = ProjectionMLP(config['stu_hidden_dim'], config['proj_dim']).to(device)
    teacher_proj_head = ProjectionMLP(config['tea_hidden_dim'], config['proj_dim']).to(device)

    # 优化器
    optimizer = Adam(
        list(student_model.parameters()) + list(student_proj_head.parameters()),
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config['patience'])

    # 训练历史
    history = {
        'epoch': [],
        'train_loss': [],
        'train_task_loss': [],
        'train_kd_loss': [],
        'train_kl_loss': [],
        'train_nce_loss': [],
        'train_core_nce_loss': [],
        'train_subgraph_loss': [],
        'train_topk_loss': [],
        'val_accuracy': [],
        'val_loss': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 创建保存目录
    save_dir = os.path.join(base_output_dir, f"run_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    history_path = os.path.join(save_dir, 'training_history.csv')

    print("\n开始 KD 训练...")
    start_time = time.time()

    for epoch in range(config['epochs']):
        train_metrics_list = []
        
        # 训练循环
        train_pbar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]', 
            leave=False, 
            file=sys.stderr
        )

        for batch in train_pbar:
            metrics = train_step(
                teacher_model, student_model, 
                student_proj_head, teacher_proj_head,
                batch, optimizer, device, config
            )
            
            train_metrics_list.append(metrics)
            train_pbar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                'Task': f"{metrics['task_loss']:.4f}",
                'KD': f"{metrics['kd_loss']:.4f}",
                'KL': f"{metrics['kl_loss']:.4f}",
                'NCE': f"{metrics['nce_loss']:.4f}",
                'Core': f"{metrics['loss_core_rkd']:.4f}",
                'SubG': f"{metrics['subgraph_loss']:.4f}",
                'TopK': f"{metrics['loss_topk']:.4f}"
            })

        # 平均训练指标
        avg_train_metrics = {
            k: float(np.mean([m[k] for m in train_metrics_list])) 
            for k in train_metrics_list[0].keys()
        }

        # 验证
        val_metrics = evaluate_model(student_model, val_loader, device)
        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_metrics['total_loss'])
        history['train_task_loss'].append(avg_train_metrics['task_loss'])
        history['train_kd_loss'].append(avg_train_metrics['kd_loss'])
        history['train_kl_loss'].append(avg_train_metrics['kl_loss'])
        history['train_nce_loss'].append(avg_train_metrics['nce_loss'])
        history['train_core_nce_loss'].append(avg_train_metrics['loss_core_rkd'])
        history['train_subgraph_loss'].append(avg_train_metrics['subgraph_loss'])
        history['train_topk_loss'].append(avg_train_metrics['loss_topk'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['lr'].append(current_lr)

        is_best = val_metrics['accuracy'] > best_val_acc

        # 定期打印
        if (epoch + 1) % 10 == 0 or is_best:
            print(
                f"Run {run_idx+1} Epoch {epoch+1:3d} | "
                f"Total: {avg_train_metrics['total_loss']:.4f} | "
                f"Task: {avg_train_metrics['task_loss']:.4f} | "
                f"KL: {avg_train_metrics['kl_loss']:.4f} | "
                f"NCE: {avg_train_metrics['nce_loss']:.4f} | "
                f"Core: {avg_train_metrics['loss_core_rkd']:.4f} | "
                f"SubG: {avg_train_metrics['subgraph_loss']:.4f} | "
                f"TopK: {avg_train_metrics['loss_topk']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

        # 保存最佳模型
        if is_best:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  >>> New Best Val Acc: {best_val_acc:.4f} (Saved)")
            best_model_state = student_model.state_dict()
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config['patience']:
            print(f"\n早停触发！在第 {epoch+1} 轮停止训练 (Best Val: {best_val_acc:.4f})")
            break

    training_time = time.time() - start_time

    # 保存训练历史
    df_history = pd.DataFrame(history)
    df_history.to_csv(history_path, index=False)

    # 加载最佳模型并测试
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
        torch.save(best_model_state, best_model_path)

    test_metrics = evaluate_model(student_model, test_loader, device)
    teacher_test_metrics = evaluate_model(teacher_model, test_loader, device)

    print(f"训练完成! 最佳验证: {best_val_acc:.4f} (epoch {best_epoch}), 用时: {training_time:.2f}s")
    print(f"学生测试准确率: {test_metrics['accuracy']:.4f}")
    print(f"教师测试准确率: {teacher_test_metrics['accuracy']:.4f}")
    print(f"训练历史保存路径: {history_path}")
    print(f"最佳模型权重保存路径: {best_model_path}")

    return test_metrics, best_val_acc, best_model_state, best_epoch


# def set_seed(seed):
#     """固定随机种子"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
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

def _parse_list_arg(value, cast_fn):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [cast_fn(v) for v in value]
    items = [v.strip() for v in str(value).split(",")]
    items = [v for v in items if v != ""]
    return [cast_fn(v) for v in items]


def parse_args():
    parser = argparse.ArgumentParser(description="NCI1 Graph-to-MLP KD Training")

    parser.add_argument("--dataset_name", type=str, default="NCI1")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--teacher_model", type=str, default="GIN")
    parser.add_argument("--tea_hidden_dim", type=int, default=128)
    parser.add_argument("--tea_layers", type=int, default=3)
    parser.add_argument("--tea_pool_type", type=str, default="add")
    parser.add_argument("--dropout_tea", type=float, default=0.2)
    parser.add_argument("--teacher_path", type=str, default="output/NCI1/GIN/seed1/best_model.pth")

    parser.add_argument("--student_model", type=str, default="MLP")
    parser.add_argument("--stu_hidden_dim", type=int, default=128)
    parser.add_argument("--stu_layers", type=int, default=3)
    parser.add_argument("--stu_pool_type", type=str, default="mean")
    parser.add_argument("--dropout_stu", type=float, default=0.2)

    parser.add_argument("--proj_dim", type=int, default=128)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)

    parser.add_argument("--task_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.9)
    parser.add_argument("--nce_weight", type=float, default=0.0)
    parser.add_argument("--core_nce_weight", type=float, default=0.005)
    parser.add_argument("--subgraph_weight", type=float, default=15.0)
    parser.add_argument("--topk_weight", type=float, default=5.0)

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--topk_k", type=int, default=4)

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    return parser.parse_args()


def build_config(args):
    config = {
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "teacher_model": args.teacher_model,
        "tea_hidden_dim": args.tea_hidden_dim,
        "tea_layers": args.tea_layers,
        "tea_pool_type": args.tea_pool_type,
        "dropout_tea": args.dropout_tea,
        "teacher_path": args.teacher_path,
        "student_model": args.student_model,
        "stu_hidden_dim": args.stu_hidden_dim,
        "stu_layers": args.stu_layers,
        "stu_pool_type": args.stu_pool_type,
        "dropout_stu": args.dropout_stu,
        "proj_dim": args.proj_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "task_weight": args.task_weight,
        "kl_weight": args.kl_weight,
        "nce_weight": args.nce_weight,
        "core_nce_weight": args.core_nce_weight,
        "subgraph_weight": args.subgraph_weight,
        "topk_weight": args.topk_weight,
        "temperature": args.temperature,
        "topk_k": args.topk_k,
        "seeds": _parse_list_arg(args.seeds, int),
    }
    return config

def main():
    args = parse_args()
    config = build_config(args)
 
    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 一次性加载数据（所有 seed 共用）
    print("加载 NCI1 数据集...")
    train_loader, val_loader, test_loader, dataset_info = load_nci1_dataset(
        batch_size=config['batch_size'],
        use_degree=True,
        pe_dim=8,
        add_spd=False,
        add_core=True,
        add_spectral=True,
        n_clusters=5,
        name=config['dataset_name']
    )
    # train_loader, val_loader, test_loader, dataset_info = load_tu_dataset(
    #     batch_size=config['batch_size'],
    #     name=config['dataset_name'],
    # )
    # 遍历不同的蒸馏权重配置（这里以 KL 权重为例）
        # 创建日志目录和文件
    log_dir = f'Experiment-{config["dataset_name"]}'
    os.makedirs(log_dir, exist_ok=True)
        
    log_file_path = (
            f'{log_dir}/M3-RKD_{config["dataset_name"]}_'
            f'{config["teacher_model"]}_{config["tea_layers"]}_dim{config["tea_hidden_dim"]}_{config["dropout_tea"]}_{config["student_model"]}{config["stu_layers"]}_dim{config["stu_hidden_dim"]}_{config["dropout_stu"]}/'
            f'pool-{config["topk_k"]}_{config["stu_pool_type"]}_kl{config["kl_weight"]}_nce{config["nce_weight"]}_core{config["core_nce_weight"]}_'
            f'sub{config["subgraph_weight"]}_topk{config["topk_weight"]}.txt'
        )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # 创建输出目录
    output_dir = f'./output/{config["dataset_name"]}_KD'
    son_dir = (
            f"M3-RKD_{config['teacher_model']}{config['tea_layers']}_to_"
            f"{config['student_model']}{config['stu_layers']}/"
            f"kl{config['kl_weight']}_core{config['core_nce_weight']}_sub{config['subgraph_weight']}_topk{config['topk_weight']}_k{config['topk_k']}"
        )
    base_output_dir = os.path.join(output_dir, son_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    original_stdout = sys.stdout

        # 重定向输出到日志文件
    with open(log_file_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            print(f"Configuration: {config}")
            print(f"Device: {device}")
            print(f"Log file: {log_file_path}")
            print(f"Output directory: {base_output_dir}")

            # 初始化结果列表
            all_test_accuracies = []
            
            # 多种子训练
            for run_idx, seed in enumerate(config['seeds']):
                set_seed(seed)
                config['seed'] = seed
                
                # 运行训练
                test_metrics, _, _, _ = train_and_evaluate_kd(
                    config, device, base_output_dir, run_idx,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    dataset_info=dataset_info
                )
                
                # 收集结果
                all_test_accuracies.append(test_metrics['accuracy'])

            # 计算统计结果
            mean_acc = np.mean(all_test_accuracies)
            std_acc = np.std(all_test_accuracies)
            
            # 输出汇总
            print(f"\n{'#'*60}")
            print(f"最终结果汇总 ({len(config['seeds'])} runs):")
            print(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"详细结果: {all_test_accuracies}")
            print(f"{'#'*60}")
            
            # 保存到汇总 CSV
            results_csv_path = os.path.join(output_dir, 'M3-RKD_summary_results.csv')
            
            summary_data = {
                'dataset': config['dataset_name'],
                'teacher': f"{config['teacher_model']}_L{config['tea_layers']}_H{config['tea_hidden_dim']}",
                'student': f"{config['student_model']}_L{config['stu_layers']}_H{config['stu_hidden_dim']}",
                'seeds_count': len(config['seeds']),
                'kl_w': config['kl_weight'],
                'nce_w': config['nce_weight'],
                'core_nce_w': config['core_nce_weight'],
                'subgraph_w': config['subgraph_weight'],
                'topk_w': config['topk_weight'],
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'all_accuracies': str(all_test_accuracies)
            }
            
            df_result = pd.DataFrame([summary_data])
            if os.path.exists(results_csv_path):
                df_result.to_csv(results_csv_path, mode='a', header=False, index=False)
            else:
                df_result.to_csv(results_csv_path, mode='w', header=True, index=False)
            
            print(f"汇总结果已保存至: {results_csv_path}")

            # 恢复标准输出
            sys.stdout = original_stdout
            print(f"Training finished. Log saved to {log_file_path}")



if __name__ == '__main__':
    main()
