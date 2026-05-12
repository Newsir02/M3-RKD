import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Optional, Tuple
from .util import n_times_k_step_random_walks, get_random_walk_nodes, batch_generate_walks, build_adj_list
import math
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax
from torch_scatter import scatter_max, scatter_add , scatter_mean
from torch_geometric.utils import to_dense_batch
def nce_criterion(student_feat, teacher_feat, nce_T=0.075, max_samples=8192):
    """Graph Contrastive Representation Distillation, [Joshi et al., TNNLS 2022](https://arxiv.org/abs/2111.04964)
    """
    if max_samples < student_feat.shape[0]:
        sampled_inds = np.random.choice(student_feat.shape[0], max_samples, replace=False)
        student_feat = student_feat[sampled_inds]
        teacher_feat = teacher_feat[sampled_inds]
    
    student_feat = F.normalize(student_feat, p=2, dim=-1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

    nce_logits = torch.mm(student_feat, teacher_feat.transpose(0, 1))  # [N, N] 学生和老师的余弦相似度
    nce_labels = torch.arange(student_feat.shape[0]).to(student_feat.device)

    nce_loss = F.cross_entropy(nce_logits/ nce_T, nce_labels)
    
    return nce_loss

def nce_criterion_grouped(student_feat: torch.Tensor,
                          teacher_feat: torch.Tensor,
                          group_ids: torch.Tensor,
                          nce_T: float = 0.075) -> torch.Tensor:
    student_feat = F.normalize(student_feat, p=2, dim=-1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
    loss_out = torch.zeros(student_feat.size(0), device=student_feat.device)
    groups = torch.unique(group_ids)
    for g in groups:
        idx = torch.nonzero(group_ids == g, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        s_g = student_feat[idx]
        t_g = teacher_feat[idx]
        logits_g = torch.mm(s_g, t_g.transpose(0, 1))
        labels_g = torch.arange(idx.size(0), device=student_feat.device)
        loss_g = F.cross_entropy(logits_g / nce_T, labels_g, reduction='none')
        loss_out[idx] = loss_g
    return loss_out

def graph_topology_attention_loss(attn_weights_list, data, sigma=2.0):
    """
    针对 MLP 稀疏注意力的优化版 Loss (接收 Logits，内部处理温度和 Softmax)
    """
    # 1. 获取 Ground Truth (物理距离)
    row, col = data.spatial_pe_idx
    dist = data.spatial_pe_val
    
    # 2. 获取 Student 的预测 (Raw Logits)
    student_logits = attn_weights_list[0]
    
    # ============================================================
    # 【关键修改】在此处处理温度 (Temperature Scaling)
    # ============================================================
    # 在蒸馏中，通常 student_logits / T。
    # 这里我们复用 sigma 作为温度参数 T。
    # sigma 越大 -> 温度越高 -> 分布越平滑 -> 关注更远的邻居
    # sigma 越小 -> 温度越低 -> 分布越尖锐 -> 只关注最近的邻居
    scaled_student_logits = student_logits / sigma
    
    # 进行稀疏 Softmax (按发送节点 row 分组归一化)
    # 这一步将 Logits (-inf, +inf) 变成了 Probabilities (0, 1)
    student_prob = softmax(scaled_student_logits, index=row, num_nodes=data.num_nodes)
    
    # 转换为 Log Probabilities (用于 KL Div 输入)
    student_log_prob = torch.log(student_prob + 1e-16)
    
    # ------------------------------------------------------------
    
    # 3. 计算 Teacher (Target) 分布
    # 公式: P_t = exp(-d^2 / 2sigma^2) / Sum
    # 注意：这里的 2*sigma^2 本质上也是一种温度缩放
    #target_score = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    target_score = torch.exp(-dist / sigma)
    # 归一化 Target
    row_sum = scatter(target_score, row, dim=0, dim_size=data.num_nodes, reduce='sum')
    target_prob = target_score / (row_sum[row] + 1e-16)

    # 4. 计算 KL 散度
    # F.kl_div(input=log_prob, target=prob)
    kl_loss = F.kl_div(student_log_prob, target_prob, reduction='sum')

    # 5. 归一化 (除以总节点数)
    return kl_loss / data.num_nodes
def kl_divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """通用KL蒸馏损失：同时支持多类与二分类（单输出）

    多类：对类别维度做 softmax/log_softmax 后计算 KL
    二分类（输出维度=1）：对 logits 做 temperature 缩放后，用 sigmoid 得到软标签，
    再以 BCE with logits 作为 KL 的替代形式（等价于 Bernoulli 分布的 KL 近似），并乘以 T^2。
    """
    s = student_logits / temperature
    t = teacher_logits / temperature

    if student_logits.dim() == 2 and student_logits.size(1) == 1:
        teacher_probs = torch.sigmoid(t)
        bce = F.binary_cross_entropy_with_logits(s, teacher_probs, reduction='mean')
        return bce * (temperature ** 2)
    else:
        teacher_probs = F.softmax(t, dim=1)
        student_log_probs = F.log_softmax(s, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl * (temperature ** 2)

def cosine_similarity_matrix(embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    计算嵌入向量之间的余弦相似度矩阵
    
    Args:
        embeddings: [N, D] 节点嵌入
        normalize: 是否对嵌入进行L2归一化
    
    Returns:
        similarity_matrix: [N, N] 余弦相似度矩阵
    """
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    return similarity_matrix

def compute_batch_random_walk_similarity(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
    adj_list: dict,  # 接收预构建的邻接表
    node_indices: torch.Tensor, # 采样的中心节点索引
    n: int, k: int,
    similarity_func: str = 'cosine'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    模块化函数：批量计算 Teacher 和 Student 在随机游走路径上的相似度矩阵
    """
    device = teacher_embeddings.device
    
    # 1. 批量生成游走路径 [Num_Samples, N, K+1]
    # 这里调用 utils 中的 batch_generate_walks
    walks = batch_generate_walks(adj_list, node_indices.tolist(), n, k, device)
    
    # 2. 提取路径节点（排除起始点） [Num_Samples, N, K]
    walk_nodes = get_random_walk_nodes(walks, exclude_start=True)
    
    # 3. 准备向量化计算 (Vectorized Computation)
    # 我们要利用广播机制一次性算出所有相似度，避免任何 Python 循环
    
    # 获取中心节点嵌入: [Num_Samples, D] -> [Num_Samples, 1, 1, D]
    t_center = teacher_embeddings[node_indices].unsqueeze(1).unsqueeze(1)
    s_center = student_embeddings[node_indices].unsqueeze(1).unsqueeze(1)
    
    # 获取游走节点嵌入: 
    # walk_nodes 是 [Num_Samples, N, K]
    # embedding 是 [Total_Nodes, D]
    # 结果 t_walk 是 [Num_Samples, N, K, D]
    t_walk = teacher_embeddings[walk_nodes] 
    s_walk = student_embeddings[walk_nodes]
    
    # 4. 计算相似度矩阵 [Num_Samples, N, K]
    if similarity_func == 'cosine':
        t_sim = F.cosine_similarity(t_center, t_walk, dim=-1)
        s_sim = F.cosine_similarity(s_center, s_walk, dim=-1)
    
    elif similarity_func == 'euclidean':
        # 负欧氏距离
        t_sim = torch.norm(t_center - t_walk, p=2, dim=-1)
        s_sim = torch.norm(s_center - s_walk, p=2, dim=-1)

    elif similarity_func == 'guss_euclidean':
        # 负欧氏距离
        t_dist = torch.norm(t_center - t_walk, p=2, dim=-1)
        s_dist = torch.norm(s_center - s_walk, p=2, dim=-1)
        
        # 使用特征维度作为 sigma 的基础，或者直接使用距离本身
        # 这里修正之前的错误：d = t_center 是错误的，t_center 是张量
        # 通常高斯核的 sigma 可以设为 sqrt(d_model) 或者固定值
        d_model = teacher_embeddings.shape[-1]
        sigma = math.sqrt(d_model)
        
        t_sim = torch.exp(-(t_dist ** 2) / (2 * sigma ** 2))
        s_sim = torch.exp(-(s_dist ** 2) / (2 * sigma ** 2))  

    elif similarity_func == 'dot':
        t_sim = (t_center * t_walk).sum(dim=-1)
        s_sim = (s_center * s_walk).sum(dim=-1)
    else:
        raise ValueError(f"Unknown similarity function: {similarity_func}")
        
    return t_sim, s_sim


def random_walk_distillation_loss(teacher_embeddings: torch.Tensor,
                                 student_embeddings: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 n: int, k: int,
                                 node_indices: Optional[torch.Tensor] = None,
                                 similarity_func: str = 'cosine',
                                 loss_type: str = 'mse',
                                 temperature: float = 1.0,
                                 threshold = 0.1) -> torch.Tensor:
    """
    基于共享随机游走路径的知识蒸馏损失
    
    Args:
        teacher_embeddings: [num_nodes, D] 教师模型节点嵌入
        student_embeddings: [num_nodes, D] 学生模型节点嵌入
        edge_index: [2, E] 边索引
        n: 随机游走次数
        k: 随机游走步数
        node_indices: 要计算损失的节点索引（通常是训练节点）
        similarity_func: 相似度函数类型
        loss_type: 损失函数类型 ('mse', 'kl', 'cosine')
        temperature: 温度参数（用于KL散度）
    
    Returns:
        loss: 蒸馏损失
    """
    device = teacher_embeddings.device
    num_total_nodes = teacher_embeddings.shape[0]

    # 处理node_indices参数，确保它是张量格式
    if node_indices is None:
        num_nodes = min(200, teacher_embeddings.shape[0])
        node_indices = torch.arange(num_nodes, device=device)
    else:
        # 如果node_indices是布尔掩码，转换为索引张量
        if node_indices.dtype == torch.bool:
            node_indices = torch.nonzero(node_indices, as_tuple=False).squeeze(1)
        
        # 确保node_indices在正确的设备上
        if node_indices.device != device:
            node_indices = node_indices.to(device)
    
    adj_list = build_adj_list(edge_index, num_total_nodes)
    
    # === 优化核心 2: 调用模块化的相似度计算函数 ===
    # 直接得到形状为 [Num_Samples, N, K] 的矩阵
    teacher_sim_matrices, student_sim_matrices = compute_batch_random_walk_similarity(
        teacher_embeddings, student_embeddings,
        adj_list, node_indices,
        n, k, similarity_func
    )
    
    # 计算损失
    if loss_type == 'mse':
        path_loss = F.mse_loss(student_sim_matrices, teacher_sim_matrices)
    
        
    elif loss_type == 'relaxed_mse':
        mask = teacher_sim_matrices > threshold
        
        if mask.sum() > 0:
            # 只计算 Mask 部分的 MSE
            path_loss = F.mse_loss(student_sim_matrices[mask], teacher_sim_matrices[mask])
        else:
            # 如果没有满足条件的（虽然不太可能），退化为普通 MSE 或返回 0
            path_loss = torch.tensor(0.0, device=device, requires_grad=True)

    elif loss_type == 'pearson':
        # [改进 2] Pearson Correlation Loss: 关注分布趋势，忽略数值绝对大小差异
        # 将矩阵展平为 [Num_Samples, -1] 以计算每个样本的分布相关性
        t_flat = teacher_sim_matrices.flatten(1)
        s_flat = student_sim_matrices.flatten(1)
        
        # 归一化 (减均值，除标准差)
        t_mean = t_flat.mean(dim=1, keepdim=True)
        s_mean = s_flat.mean(dim=1, keepdim=True)
        t_std = t_flat.std(dim=1, keepdim=True) + 1e-8
        s_std = s_flat.std(dim=1, keepdim=True) + 1e-8
        
        t_norm = (t_flat - t_mean) / t_std
        s_norm = (s_flat - s_mean) / s_std
        
        # 计算相关系数 (期望是1，所以 loss = 1 - corr)
        correlation = (t_norm * s_norm).mean(dim=1)
        path_loss = 1 - correlation.mean()
    elif loss_type == 'kl':
        # 使用温度缩放的KL散度
        teacher_probs = F.softmax(teacher_sim_matrices/ temperature, dim=-1)
        student_log_probs = F.log_softmax(student_sim_matrices / temperature, dim=-1)
        path_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        path_loss *= temperature ** 2
        # path_loss = path_loss / (n*k)
    elif loss_type == 'cosine':
        # 余弦相似度损失
        teacher_flat = teacher_sim_matrices.flatten(1)
        student_flat = student_sim_matrices.flatten(1)
        path_loss = 1 - F.cosine_similarity(teacher_flat, student_flat, dim=1).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return path_loss


def compute_core_view_graph_emb(node_emb, batch, core_indices, t=2):
    """
    修正版：加入归一化，消除 GIN 和 MLP 特征模长差异对 Attention 分布的影响
    """
    core_emb = node_emb[core_indices].unsqueeze(-1)

    nodes_dense, mask = to_dense_batch(node_emb, batch)

    nodes_dense_norm = F.normalize(nodes_dense, p=2, dim=-1)
    core_emb_norm = F.normalize(core_emb, p=2, dim=1) 
    sim_score = torch.bmm(nodes_dense_norm, core_emb_norm).squeeze(-1)

    sim_score[~mask] = -1e9
    alpha = F.softmax(sim_score/t, dim=1)
    
    core_view_graph_emb = torch.bmm(nodes_dense.transpose(1, 2), alpha.unsqueeze(-1)).squeeze(-1)

     
    sim = F.cosine_similarity(core_view_graph_emb, core_emb.squeeze(-1), dim=1)

    return core_view_graph_emb

def rkd_distance_loss(teacher_core_emb, student_core_emb, delta=1.0):
    """
    Relational Knowledge Distillation (RKD) - Distance-wise Loss
    Paper: Park et al., "Relational Knowledge Distillation", CVPR 2019
    
    Args:
        teacher_feat: [Batch, Dim]
        student_feat: [Batch, Dim]
        delta: Huber Loss 的阈值参数，论文中通常设为 1.0 或 2.0
    """
    # 1. 计算成对欧氏距离 (Pairwise Euclidean Distance)
    # pdist: [Batch, Batch]
    # t_norm = F.normalize(teacher_core_emb, p=2,dim=-1)
    # s_norm = F.normalize(student_core_emb, p=2,dim=-1)

    t_dist = torch.cdist(teacher_core_emb, teacher_core_emb, p=2)
    s_dist = torch.cdist(student_core_emb, student_core_emb, p=2)

    # 2. 距离归一化 (Normalization)
    # 论文核心：除以平均距离，使 Loss 具有 Scale Invariance。
    # 只计算非零部分（排除对角线），避免 0 影响均值计算
    t_mean = t_dist[t_dist > 0].mean()
    s_mean = s_dist[s_dist > 0].mean()

    # 加上 1e-8 防止除以 0
    t_dist_norm = t_dist / (t_mean + 1e-8)
    s_dist_norm = s_dist / (s_mean + 1e-8)

    # 3. Huber Loss
    # 强制让学生去拟合老师的归一化距离矩阵
    # F.huber_loss 在 PyTorch 1.9+ 可用，比 smooth_l1_loss 更符合论文语义
    loss = F.huber_loss(s_dist_norm, t_dist_norm, reduction='mean', delta=delta)
    
    return loss

def subgraph_distillation_loss(teacher_node_emb, student_node_emb, 
                               batch_fragment_index, batch_graph_index, loss_type='kl', temperature=1.0
                               ):
    """
    基于 BRICS 子图的蒸馏 Loss (完全向量化版本，无循环)
    """

    # [Total_Fragments, D]
    t_sub = scatter_add(teacher_node_emb, batch_fragment_index, dim=0)
    s_sub = scatter_add(student_node_emb, batch_fragment_index, dim=0)
    
    # Loss 1: 特征直接对齐
    loss_feat = torch.tensor(0.0, device=teacher_node_emb.device)
    #loss_feat = F.mse_loss(s_sub, t_sub)

    # 2. 准备 Batch 索引
    frag_to_graph = scatter_mean(batch_graph_index.float(), batch_fragment_index, dim=0).long()

    # 3. 转为稠密矩阵 (To Dense)
    # 将堆叠的片段转换为 [Batch_Size, Max_Frags_Per_Graph, Dim]
    t_dense, mask = to_dense_batch(t_sub, frag_to_graph)
    s_dense, _    = to_dense_batch(s_sub, frag_to_graph)

    t_norm = F.normalize(t_dense, p=2, dim=-1, eps=1e-8)
    s_norm = F.normalize(s_dense, p=2, dim=-1, eps=1e-8)

    t_sim_matrix = torch.bmm(t_norm, t_norm.transpose(1, 2))
    s_sim_matrix = torch.bmm(s_norm, s_norm.transpose(1, 2))
    # valid_matrix_mask[b, i, j] = True 当且仅当 片段i 和 片段j 都是真实的
    valid_matrix_mask = mask.unsqueeze(2) & mask.unsqueeze(1)

    B, M, _ = valid_matrix_mask.shape
    # 生成单位矩阵 (对角线为1)，然后取反 (对角线为0)
    diag_mask = ~torch.eye(M, device=valid_matrix_mask.device).bool().unsqueeze(0)
    
    # 最终 Mask: 既要是真实存在的片段，又不能是对角线
    final_mask = valid_matrix_mask & diag_mask

    if final_mask.sum() > 0:
        if loss_type == 'kl':
            temperature = temperature
            t_logits = t_sim_matrix.clone()
            s_logits = s_sim_matrix.clone()
            
            t_logits = t_logits.masked_fill(~final_mask, -1e9)
            s_logits = s_logits.masked_fill(~final_mask, -1e9)
            
            t_prob = F.softmax(t_logits / temperature, dim=-1)
            s_log_prob = F.log_softmax(s_logits / temperature, dim=-1)
            
            kl_map = F.kl_div(s_log_prob, t_prob, reduction='none')

            loss_rel = (kl_map * final_mask.float()).sum() / final_mask.sum()
        else:
            loss_rel = F.mse_loss(s_sim_matrix[final_mask], 
                              t_sim_matrix[final_mask])
    else:
        loss_rel = torch.tensor(0.0, device=teacher_node_emb.device)
        
    return loss_feat, loss_rel

def topk_ranking_distillation_loss(
    teacher_emb: torch.Tensor, 
    student_emb: torch.Tensor, 
    batch: torch.Tensor, 
    k: int = 5, 
    temperature: float = 1.0,
    similarity_metric: str = 'cosine'
):
    """
    Top-K 节点重要性排序蒸馏 (Inspired by List-wise Ranking in RecSys)
    
    Args:
        teacher_emb: [Total_Nodes, D]
        student_emb: [Total_Nodes, D]
        batch: [Total_Nodes]
        k: 关注前 K 个最重要的节点
        temperature: 温度系数，控制分布的锐度
    """
    # 1. 转换为 Dense Batch [Batch, Max_Nodes, D]
    t_dense, mask = to_dense_batch(teacher_emb, batch)
    s_dense, _    = to_dense_batch(student_emb, batch)
    
    # 2. 计算相似度矩阵 (Attention Map) [Batch, Max_Nodes, Max_Nodes]
    # 使用 Cosine Similarity
    if similarity_metric == 'cosine':
        t_dense = F.normalize(t_dense, p=2, dim=-1)
        s_dense = F.normalize(s_dense, p=2, dim=-1)
        t_sim = torch.bmm(t_dense, t_dense.transpose(1, 2))
        s_sim = torch.bmm(s_dense, s_dense.transpose(1, 2))
    else: # Dot product
        t_sim = torch.bmm(t_dense, t_dense.transpose(1, 2))
        s_sim = torch.bmm(s_dense, s_dense.transpose(1, 2))

    # 3. Mask 处理
    # 创建有效位置的 Mask [Batch, N, N]
    valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    
    # 排除对角线 (自己对自己总是最重要的，没有学习意义)
    B, N, _ = valid_mask.shape
    diag_mask = ~torch.eye(N, device=teacher_emb.device).bool().unsqueeze(0)
    final_mask = valid_mask & diag_mask
    
    # 将无效位置设为极小值 (-inf)，确保 Top-K 不会选中 padding 或自己
    t_sim = t_sim.masked_fill(~final_mask, -1e9)
    s_sim = s_sim.masked_fill(~final_mask, -1e9)

    # 4. 获取 Teacher 的 Top-K 索引
    # 我们希望 Student 在 Teacher 认为最重要的 K 个节点上的分布与 Teacher 一致
    # 实际 K 不能超过当前图的大小，取 min
    real_k = min(k, N - 1) 
    
    # values_t: [Batch, N, K], indices_t: [Batch, N, K]
    # 这里的 indices_t 就是对于每个节点 i，Teacher 认为最重要的 K 个邻居的索引
    t_topk_val, t_topk_idx = torch.topk(t_sim, k=real_k, dim=-1)
    
    # 5. 从 Student 的相似度矩阵中，提取相同位置的数值
    # gather 的维度 dim=2，意味着在最后一个维度(目标节点维度)按 Teacher 的索引取值
    s_selected_val = torch.gather(s_sim, 2, t_topk_idx)
    
    # 6. 计算 List-wise KL Divergence (ListNet Loss)
    # 对这 K 个值进行 Softmax 归一化，变成概率分布
    # 含义：在最重要的 K 个邻居中，Student 应该保持和 Teacher 一样的相对重要性比例
    t_prob = F.softmax(t_topk_val / temperature, dim=-1)
    s_log_prob = F.log_softmax(s_selected_val / temperature, dim=-1)
    
    # KL Div: 目标是 Teacher，输入是 Student (Log Prob)
    # reduction='none' 得到 [Batch, N]，需要在 mask 上取平均
    loss_map = F.kl_div(s_log_prob, t_prob, reduction='none').sum(dim=-1)
    
    # 只计算真实节点的 Loss 平均值
    loss = (loss_map * mask).sum() / (mask.sum() + 1e-8)
    
    return loss

# NOSMOG   NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs
def nosmog_rsd_loss(teacher_node_emb:  torch.Tensor, 
                    student_node_emb: torch.Tensor, 
                    batch: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
    """
    NOSMOG 的表征相似性蒸馏 (Representation Similarity Distillation)
    """
    # 1. 归一化嵌入（确保相似性在 [-1, 1] 范围内）
    t_norm = F.normalize(teacher_node_emb, p=2, dim=-1)
    s_norm = F.normalize(student_node_emb, p=2, dim=-1)
    
    # 2. 转换为稠密批次 [Batch_Size, Max_Nodes_Per_Graph, Hidden_Dim]
    # 这一步会自动对齐每个图的节点数（padding）
    t_dense, mask = to_dense_batch(t_norm, batch)
    s_dense, _    = to_dense_batch(s_norm, batch)
    
    # 3. 计算批内相似性矩阵 [Batch_Size, Max_Nodes, Max_Nodes]
    # 使用批量矩阵乘法（超级快！）
    t_sim_matrix = torch.bmm(t_dense, t_dense.transpose(1, 2)) / temperature
    s_sim_matrix = torch.bmm(s_dense, s_dense.transpose(1, 2)) / temperature
    
    # 4. 创建有效矩阵掩码
    # 只计算真实节点之间的相似性（忽略 padding）
    # mask:  [Batch, Max_Nodes] -> [Batch, Max_Nodes, Max_Nodes]
    valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    
    # 5. 排除对角线（节点与自己的相似性总是1，没有学习价值）
    batch_size, max_nodes, _ = valid_mask.shape
    diag_mask = ~torch.eye(max_nodes, device=valid_mask.device).bool().unsqueeze(0)
    final_mask = valid_mask & diag_mask
    
    # 6. 只计算有效位置的 MSE Loss
    if final_mask.sum() > 0:
        loss = F.mse_loss(s_sim_matrix[final_mask], t_sim_matrix[final_mask])
    else:
        loss = torch.tensor(0.0, device=teacher_node_emb.device)
    
    return loss
# NOSMOG   NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs
def nosmog_adv_loss(student_model, batch, graph_emb:  torch.Tensor, 
                   labels: torch.Tensor, criterion,
                   eps: float = 0.3, alpha: float = 0.075, iters: int = 5) -> torch.Tensor:
    """
    NOSMOG 对抗训练损失（PGD 攻击图级嵌入）

    """
    # 1. 初始化随机扰动 δ ∈ [-eps, eps]
    delta = (torch.rand_like(graph_emb) * 2 - 1) * eps
    delta.requires_grad = True
    
    # 2. PGD 迭代优化（梯度上升，最大化损失）
    for _ in range(iters):
        # 加扰动
        perturbed_emb = graph_emb + delta
        
        # 从图嵌入到预测（只经过分类头）
        if hasattr(student_model, 'classifier'):
            logits = student_model.classifier(perturbed_emb)
        elif hasattr(student_model, 'pred_layer'):
            logits = student_model.pred_layer(perturbed_emb)
        else:
            raise AttributeError("Student model must have 'classifier' or 'pred_layer'")
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播（只更新 delta，不更新模型）
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        # 更新 delta：δ ← δ + α·sign(∇_δ L)
        delta.data = delta.data + alpha * delta.grad.sign()
        
        # 投影到 L∞ 球：δ ← clip(δ, -eps, eps)
        delta.data = torch.clamp(delta.data, -eps, eps)
    
    # 3. 计算最终对抗样本的损失（detach delta，不再优化）
    with torch.no_grad():
        adv_emb = graph_emb + delta.detach()
        if hasattr(student_model, 'classifier'):
            adv_logits = student_model.classifier(adv_emb)
        elif hasattr(student_model, 'pred_layer'):
            adv_logits = student_model.pred_layer(adv_emb)
        
        adv_loss = criterion(adv_logits, labels)
    
    return adv_loss

def nosmog_adv_loss_node_level(student_model, batch, 
                               criterion,
                               eps:  float = 0.3, 
                               alpha: float = 0.075, 
                               iters: int = 5) -> torch.Tensor:
    """
    NOSMOG 对抗训练损失（节点级扰动，完全适配 StudentMLP_with_PE）
    """
    device = batch.x.device
    
    # ========== 1. 准备标签（mask NaN） ==========
    y = batch.y.view(-1, 1).float().to(device)
    mask = ~torch.isnan(y)
    
    # 如果整个 batch 都是 NaN，直接返回 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # ========== 2. 获取节点嵌入（在 node_emb 之后，PE 注入之前） ==========
    with torch.no_grad():
        # 2.1 通过 node_emb（AtomEncoder 或 Linear）
        if student_model.useEmbedding:
            x_emb = student_model.atom_encoder(batch.x.long())
        else:
            x_emb = student_model.input_proj(batch.x.float())
        
        # 2.2 PE 注入（如果启用）
        if student_model.pe_dim > 0 and student_model.add_pe and hasattr(batch, 'laplacian_pe'):
            pe = batch.laplacian_pe
            # 注意：这里不做 sign_flip（对抗训练时保持一致性）
            pe_encoded = student_model.pe_encoder(pe)
            pe_encoded = student_model.pe_norm(pe_encoded)
            x_emb = x_emb + pe_encoded
    
    # ========== 3. 初始化扰动 δ ∈ [-eps, eps] ==========
    delta = (torch.rand_like(x_emb) * 2 - 1) * eps
    delta. requires_grad = True
    
    # ========== 4. PGD 迭代（梯度上升） ==========
    for iter_idx in range(iters):
        # 4.1 加扰动
        perturbed_x = x_emb + delta
        
        # 4.2 通过 MLP layers
        x = perturbed_x
        for layer in student_model.node_mlps:
            x = layer(x)  # 每个 layer 是 Sequential(Linear -> BN -> ReLU -> Dropout)
        
        # 4.3 保存节点嵌入
        node_emb = x
        
        # 4.4 图级池化
        graph_emb = student_model.pool(node_emb, batch.batch)
        #graph_emb = student_model. dropout(graph_emb)
        
        # 4.5 分类
        logits = student_model.graph_mlp(graph_emb)
        
        # 4.6 计算损失（mask NaN）
        loss = criterion(logits[mask], y[mask])

        # 4.7 反向传播（只更新 delta）
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        # 4.8 更新 delta：δ ← δ + α·sign(∇_δ L)
        delta.data = delta.data + alpha * delta.grad.sign()
        
        # 4.9 投影到 L∞ 球：δ ← clip(δ, -eps, eps)
        delta.data = torch.clamp(delta.data, -eps, eps)
    
    # ========== 5. 计算最终对抗损失 ==========
    with torch.no_grad():
        # 5.1 最终扰动后的节点嵌入
        adv_x = x_emb + delta. detach()
        
        # 5.2 通过 MLP layers（不使用 dropout）
        x = adv_x
        for layer in student_model.node_mlps:
            # 注意：这里我们手动处理，跳过 Dropout
            # 因为 eval 模式下 Dropout 自动关闭，但我们在 train 模式
            # 所以需要临时关闭 dropout
            for sublayer in layer:
                if not isinstance(sublayer, nn.Dropout):
                    x = sublayer(x)
        
        # 5.3 池化
        graph_emb = student_model.pool(x, batch. batch)
        # 这里也不用 dropout
        
        # 5.4 分类
        adv_logits = student_model.graph_mlp(graph_emb)
        
        # 5.5 计算对抗损失（mask NaN）
        adv_loss = criterion(adv_logits[mask], y[mask])
    
    return adv_loss
# NOSMOG   NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs
def nosmog_rsd_loss(teacher_node_emb:  torch.Tensor, 
                    student_node_emb: torch.Tensor, 
                    batch: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
    """
    NOSMOG 的表征相似性蒸馏 (Representation Similarity Distillation)
    """
    from torch_geometric.utils import to_dense_batch
    # 1. 归一化嵌入（确保相似性在 [-1, 1] 范围内）
    t_norm = F.normalize(teacher_node_emb, p=2, dim=-1)
    s_norm = F.normalize(student_node_emb, p=2, dim=-1)
    
    # 2. 转换为稠密批次 [Batch_Size, Max_Nodes_Per_Graph, Hidden_Dim]
    t_dense, mask = to_dense_batch(t_norm, batch)
    s_dense, _    = to_dense_batch(s_norm, batch)
    
    # 3. 计算批内相似性矩阵 [Batch_Size, Max_Nodes, Max_Nodes]
    t_sim_matrix = torch.bmm(t_dense, t_dense.transpose(1, 2)) / temperature
    s_sim_matrix = torch.bmm(s_dense, s_dense.transpose(1, 2)) / temperature
    
    # 4. 创建有效矩阵掩码
    valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    
    # 5. 排除对角线
    batch_size, max_nodes, _ = valid_mask.shape
    diag_mask = ~torch.eye(max_nodes, device=valid_mask.device).bool().unsqueeze(0)
    final_mask = valid_mask & diag_mask
    
    # 6. 只计算有效位置的 MSE Loss
    if final_mask.sum() > 0:
        loss = F.mse_loss(s_sim_matrix[final_mask], t_sim_matrix[final_mask])
    else:
        loss = torch.tensor(0.0, device=teacher_node_emb.device)
    
    return loss

def nosmog_adv_loss(student_model, batch, graph_emb:  torch.Tensor, 
                   labels: torch.Tensor, criterion,
                   eps: float = 0.3, alpha: float = 0.075, iters: int = 5) -> torch.Tensor:
    """
    NOSMOG 对抗训练损失（PGD 攻击图级嵌入）
    """
    # 1. 初始化随机扰动 δ ∈ [-eps, eps]
    delta = (torch.rand_like(graph_emb) * 2 - 1) * eps
    delta.requires_grad = True
    
    # 2. PGD 迭代优化
    for _ in range(iters):
        perturbed_emb = graph_emb + delta
        
        if hasattr(student_model, 'classifier'):
            logits = student_model.classifier(perturbed_emb)
        elif hasattr(student_model, 'pred_layer'):
            logits = student_model.pred_layer(perturbed_emb)
        else:
            raise AttributeError("Student model must have 'classifier' or 'pred_layer'")
        
        loss = criterion(logits, labels)
        
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
    
    with torch.no_grad():
        adv_emb = graph_emb + delta.detach()
        if hasattr(student_model, 'classifier'):
            adv_logits = student_model.classifier(adv_emb)
        elif hasattr(student_model, 'pred_layer'):
            adv_logits = student_model.pred_layer(adv_emb)
        
        adv_loss = criterion(adv_logits, labels)
    
    return adv_loss

def nosmog_adv_loss_node_level(student_model, batch, 
                               criterion,
                               eps:  float = 0.3, 
                               alpha: float = 0.075, 
                               iters: int = 5) -> torch.Tensor:
    """
    NOSMOG 对抗训练损失（节点级扰动）
    自动适配 OGB (binary/multilabel, masked) 和 Multiclass (PROTEINS)
    """
    device = batch.x.device
    
    # 判断任务类型
    # PROTEINS: y is [N], long
    # OGB: y is [N, 1], float/long, contains NaNs
    is_multiclass = (batch.y.dim() == 1)
    
    if is_multiclass:
        y = batch.y.to(device)
        mask = None
    else:
        y = batch.y.view(-1, 1).float().to(device)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

    with torch.no_grad():
        if student_model.useEmbedding:
            x_emb = student_model.atom_encoder(batch.x.long())
        else:
            x_emb = student_model.input_proj(batch.x.float())
        
        if student_model.pe_dim > 0 and student_model.add_pe and hasattr(batch, 'laplacian_pe'):
            pe = batch.laplacian_pe
            pe_encoded = student_model.pe_encoder(pe)
            pe_encoded = student_model.pe_norm(pe_encoded)
            x_emb = x_emb + pe_encoded
    
    delta = (torch.rand_like(x_emb) * 2 - 1) * eps
    delta.requires_grad = True
    
    for iter_idx in range(iters):
        perturbed_x = x_emb + delta
        x = perturbed_x
        for layer in student_model.node_mlps:
            x = layer(x)
        
        node_emb = x
        graph_emb = student_model.pool(node_emb, batch.batch)
        logits = student_model.graph_mlp(graph_emb)
        
        if is_multiclass:
            loss = criterion(logits, y)
        else:
            loss = criterion(logits[mask], y[mask])

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
    
    with torch.no_grad():
        adv_x = x_emb + delta.detach()
        x = adv_x
        for layer in student_model.node_mlps:
            for sublayer in layer:
                if not isinstance(sublayer, nn.Dropout):
                    x = sublayer(x)
        
        graph_emb = student_model.pool(x, batch.batch)
        adv_logits = student_model.graph_mlp(graph_emb)
        
        if is_multiclass:
            adv_loss = criterion(adv_logits, y)
        else:
            adv_loss = criterion(adv_logits[mask], y[mask])
    
    return adv_loss



def lsp_distillation_loss(
    teacher_node_emb: torch.Tensor,
    student_node_emb: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    temperature: float = 100.0,
    similarity_type: str = 'gaussian'
) -> torch.Tensor:
    """
    Local Structure Preserving (LSP) 蒸馏损失
    来源: "Distilling Knowledge From Graph Convolutional Networks" (Yang et al., CVPR 2020)
    
    核心思想：
    1. 对每条边计算源节点和目标节点特征的相似度（使用高斯核或余弦相似度）
    2. 对每个节点的所有出边进行 softmax 归一化，得到局部结构分布
    3. 使用 KL 散度对齐 Teacher 和 Student 的局部结构分布
    
    Args:
        teacher_node_emb: [Total_Nodes, D] 教师节点嵌入
        student_node_emb: [Total_Nodes, D] 学生节点嵌入
        edge_index: [2, E] 边索引
        batch: [Total_Nodes] 批次索引
        temperature: 高斯核的温度参数（论文中为 100）
        similarity_type: 'gaussian' 或 'cosine'
    
    Returns:
        loss: LSP 蒸馏损失
    """
    device = teacher_node_emb.device
    src, dst = edge_index
    
    # ========== 1. 计算边上的相似度分数 ==========
    if similarity_type == 'gaussian':
        # 高斯距离核: exp(-||f_i - f_j||_1 / temperature)
        # 论文中使用 L1 距离（sum of absolute differences）
        
        # Teacher 相似度
        t_diff = torch.abs(teacher_node_emb[src] - teacher_node_emb[dst])  # [E, D]
        t_dist = torch.sum(t_diff, dim=-1)  # [E]
        t_score = torch.exp(-t_dist / temperature)  # [E]
        
        # Student 相似度
        s_diff = torch.abs(student_node_emb[src] - student_node_emb[dst])  # [E, D]
        s_dist = torch.sum(s_diff, dim=-1)  # [E]
        s_score = torch.exp(-s_dist / temperature)  # [E]
        
    elif similarity_type == 'cosine':
        # 余弦相似度（替代方案，更稳定但不是原论文方法）
        t_score = F.cosine_similarity(
            teacher_node_emb[src], 
            teacher_node_emb[dst], 
            dim=-1
        )  # [E]
        s_score = F.cosine_similarity(
            student_node_emb[src], 
            student_node_emb[dst], 
            dim=-1
        )  # [E]
        
        # 将 [-1, 1] 映射到 [0, 1]
        t_score = (t_score + 1) / 2
        s_score = (s_score + 1) / 2
    else:
        raise ValueError(f"Unsupported similarity_type: {similarity_type}")
    
    # ========== 2. 对每个源节点的所有出边进行 Softmax 归一化 ==========
    # 这一步将分数转换为概率分布（表示每条边在局部结构中的重要性）
    
    # Teacher 分布
    t_prob = softmax(t_score, index=src, num_nodes=teacher_node_emb.size(0))  # [E]
    
    # Student 分布（用于 KL 散度的输入需要 log）
    s_log_prob = F.log_softmax(
        torch.log(s_score + 1e-16).unsqueeze(0),  # 添加 batch 维度
        dim=0
    ).squeeze(0)
    
    # 更稳定的做法：使用 PyG 的 softmax
    s_prob_raw = softmax(s_score, index=src, num_nodes=student_node_emb.size(0))
    s_log_prob = torch.log(s_prob_raw + 1e-16)  # [E]
    
    # ========== 3. 计算 KL 散度 ==========
    # KL(Teacher || Student) = sum(p_t * log(p_t / p_s))
    # F.kl_div 需要输入为 log_prob (student) 和 prob (teacher)
    
    kl_loss = F.kl_div(s_log_prob, t_prob, reduction='sum')
    
    # ========== 4. 归一化 ==========
    # 按节点数归一化（论文做法）
    num_nodes = teacher_node_emb.size(0)
    kl_loss = kl_loss / num_nodes
    
    return kl_loss


"""
SA-MLP Style Distillation Loss for Graph-Level Classification
=============================================================
Based on: "SA-MLP: Distilling Graph Knowledge from GNNs into Structure-Aware MLP"
Adapted for graph-level classification tasks (OGB datasets).

Core idea:
  L_total = α * L_pred + β * L_node_struct + γ * L_graph_struct + δ * L_feat_align

Where:
  - L_pred:        KL divergence on logits (prediction-level distillation)
  - L_node_struct: Intra-graph node-pair similarity alignment (structure-level)
  - L_graph_struct: Inter-graph similarity alignment in batch (graph-level)
  - L_feat_align:  Direct feature alignment via MSE (feature-level)
"""

# ============================================================
#  2. Structure-Level Distillation: Intra-Graph Node Relations
# ============================================================
def sa_mlp_node_structure_loss(
    teacher_node_emb: torch.Tensor,
    student_node_emb: torch.Tensor,
    batch: torch.Tensor,
    max_nodes_per_graph: int = 128,
    loss_type: str = 'mse',
    temperature: float = 1.0,
) -> torch.Tensor:

    if teacher_node_emb.size(1) != student_node_emb.size(1):
        # 这种情况需要外部传入对齐后的 emb，这里只做一个安全检查
        raise ValueError(
            f"Teacher emb dim ({teacher_node_emb.size(1)}) != "
            f"Student emb dim ({student_node_emb.size(1)}). "
            f"Please project them to the same dimension first."
        )

    # 1. 转为 Dense Batch: [B, Max_N, D]
    t_dense, mask = to_dense_batch(teacher_node_emb, batch)
    s_dense, _    = to_dense_batch(student_node_emb, batch)

    B, N, D = t_dense.shape

    # 2. 大图采样 (防止 N^2 爆���存)
    if N > max_nodes_per_graph:
        idx = torch.randperm(N, device=t_dense.device)[:max_nodes_per_graph]
        t_dense = t_dense[:, idx, :]
        s_dense = s_dense[:, idx, :]
        mask = mask[:, idx]
        N = max_nodes_per_graph

    # 3. L2 归一化 -> 余弦相似度矩阵
    t_norm = F.normalize(t_dense, p=2, dim=-1, eps=1e-8)
    s_norm = F.normalize(s_dense, p=2, dim=-1, eps=1e-8)

    # [B, N, N] 余弦相似度矩阵
    t_sim = torch.bmm(t_norm, t_norm.transpose(1, 2))
    s_sim = torch.bmm(s_norm, s_norm.transpose(1, 2))

    # 4. 构建有效 Mask：排除 padding 位置和对角线 (自身相似度无信息量)
    valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
    diag_mask = ~torch.eye(N, device=t_dense.device).bool().unsqueeze(0)
    final_mask = valid_mask & diag_mask

    if final_mask.sum() == 0:
        return torch.tensor(0.0, device=teacher_node_emb.device)

    # 5. 计算损失
    if loss_type == 'kl':
        # KL 模式：将相似度矩阵视为"关系分布"，用 KL 散度对齐
        t_logits = t_sim.masked_fill(~final_mask, -1e9)
        s_logits = s_sim.masked_fill(~final_mask, -1e9)

        t_prob = F.softmax(t_logits / temperature, dim=-1)
        s_log_prob = F.log_softmax(s_logits / temperature, dim=-1)

        kl_map = F.kl_div(s_log_prob, t_prob, reduction='none')
        loss = (kl_map * final_mask.float()).sum() / final_mask.sum()

    else:  # 'mse' — 原论文默认方式
        # MSE 模式：直接对齐相似度值
        diff = (s_sim - t_sim) ** 2
        loss = (diff * final_mask.float()).sum() / final_mask.sum()

    return loss


# ============================================================
#  3. Structure-Level Distillation: Inter-Graph Relations
# ============================================================
def sa_mlp_graph_structure_loss(
    teacher_graph_emb: torch.Tensor,
    student_graph_emb: torch.Tensor,
    loss_type: str = 'mse',
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    图间关系蒸馏：对齐 batch 内图级嵌入之间的成对相似度矩阵。

    这是 SA-MLP 节点级思想在图级任务上的自然推广：
    如果两个图在 teacher 眼中表示相似，student 也应该给出相似的表示。

    Args:
        teacher_graph_emb: [B, D] 教师图级嵌入
        student_graph_emb: [B, D] 学生图级嵌入
        loss_type: 'mse' 或 'kl'
        temperature: KL 模式下的温度系数
    Returns:
        graph-level structure distillation loss (scalar)
    """
    B = teacher_graph_emb.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=teacher_graph_emb.device)

    # 1. L2 归一化 -> 余弦相似度
    t_norm = F.normalize(teacher_graph_emb, p=2, dim=-1, eps=1e-8)
    s_norm = F.normalize(student_graph_emb, p=2, dim=-1, eps=1e-8)

    t_sim = torch.mm(t_norm, t_norm.t())  # [B, B]
    s_sim = torch.mm(s_norm, s_norm.t())  # [B, B]

    # 2. 排除对角线
    diag_mask = ~torch.eye(B, device=teacher_graph_emb.device).bool()

    if loss_type == 'kl':
        t_logits = t_sim.masked_fill(~diag_mask, -1e9)
        s_logits = s_sim.masked_fill(~diag_mask, -1e9)

        t_prob = F.softmax(t_logits / temperature, dim=-1)
        s_log_prob = F.log_softmax(s_logits / temperature, dim=-1)

        kl_map = F.kl_div(s_log_prob, t_prob, reduction='none')
        loss = (kl_map * diag_mask.float()).sum() / diag_mask.sum()
    else:
        diff = (s_sim - t_sim) ** 2
        loss = (diff * diag_mask.float()).sum() / diag_mask.sum()

    return loss


# ============================================================
#  4. Feature-Level Alignment (直接特征对齐)
# ============================================================
def sa_mlp_feature_alignment_loss(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
    align_type: str = 'mse',
) -> torch.Tensor:
    """
    特征层对齐：直接拉近 student 和 teacher 的嵌入。

    SA-MLP 论文中解耦编码器的 H_A (结构分支) 和 H_X (特征分支) 各自
    需要与 teacher 对应的嵌入对齐。在图级任务中，我们对齐图级嵌入。

    Args:
        teacher_emb: [*, D] 教师嵌入 (节点级或图级)
        student_emb: [*, D] 学生嵌入
        align_type: 'mse' (L2) 或 'cosine' (余弦距离)
    Returns:
        feature alignment loss (scalar)
    """
    if align_type == 'cosine':
        # 余弦距离 = 1 - cos_sim
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        loss = (1 - cos_sim).mean()
    else:
        # MSE
        loss = F.mse_loss(student_emb, teacher_emb)
    return loss

def sa_mlp_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_node_emb: torch.Tensor,
    student_node_emb: torch.Tensor,
    teacher_graph_emb: torch.Tensor,
    student_graph_emb: torch.Tensor,
    batch: torch.Tensor,
    # --- 权重 ---
    pred_weight: float = 1.0,
    node_struct_weight: float = 1.0,
    graph_struct_weight: float = 0.5,
    feat_align_weight: float = 0.0,
    # --- 超参数 ---
    temperature: float = 4.0,
    struct_temperature: float = 1.0,
    struct_loss_type: str = 'mse',
    max_nodes_per_graph: int = 128,
    feat_align_type: str = 'mse',
) -> dict:

    device = student_logits.device

    # ---- 1. Prediction-Level Loss ----
    loss_pred = torch.tensor(0.0, device=device)
    if pred_weight > 0:
        loss_pred = kl_divergence_loss(
            student_logits, teacher_logits, temperature
        )

    # ---- 2. Node-Level Structure Loss (图内) ----
    loss_node_struct = torch.tensor(0.0, device=device)
    if node_struct_weight > 0:
        loss_node_struct = sa_mlp_node_structure_loss(
            teacher_node_emb, student_node_emb, batch,
            max_nodes_per_graph=max_nodes_per_graph,
            loss_type=struct_loss_type,
            temperature=struct_temperature,
        )

    # ---- 3. Graph-Level Structure Loss (图间) ----
    loss_graph_struct = torch.tensor(0.0, device=device)
    if graph_struct_weight > 0:
        loss_graph_struct = sa_mlp_graph_structure_loss(
            teacher_graph_emb, student_graph_emb,
            loss_type=struct_loss_type,
            temperature=struct_temperature,
        )

    # ---- 4. Feature Alignment Loss ----
    loss_feat_align = torch.tensor(0.0, device=device)
    if feat_align_weight > 0:
        # 图级特征对齐
        loss_feat_align = sa_mlp_feature_alignment_loss(
            teacher_graph_emb, student_graph_emb,
            align_type=feat_align_type,
        )

    # ---- Total ----
    total_loss = (
        pred_weight * loss_pred
        + node_struct_weight * loss_node_struct
        + graph_struct_weight * loss_graph_struct
        + feat_align_weight * loss_feat_align
    )

    return {
        'total_loss': total_loss,
        'pred_loss': loss_pred,
        'node_struct_loss': loss_node_struct,
        'graph_struct_loss': loss_graph_struct,
        'feat_align_loss': loss_feat_align,
    }



def sale_deepwalk_loss(student_node_emb: torch.Tensor, 
                       edge_index: torch.Tensor, 
                       walks_per_node: int = 2, 
                       walk_length: int = 3, 
                       num_neg_samples: int = 3) -> torch.Tensor:
    """
    SALE-MLP 的无监督结构损失 (DeepWalk Loss)
    通过随机游走生成正负样本对，使用 InfoNCE 对比损失让 MLP 的隐层空间学习图拓扑结构。
    
    Args:
        student_node_emb: [Total_Nodes, D] 学生模型的节点嵌入 (对应论文中的 f_theta)
        edge_index: [2, E] 当前 Batch 的边索引
        walks_per_node: 每个节点起始的随机游走次数 (对应 gamma)
        walk_length: 游走长度 (对应 window size t)
        num_neg_samples: 负采样数量 (对应 k)
        
    Returns:
        loss: 标量 Tensor
    """
    num_nodes = student_node_emb.size(0)
    device = student_node_emb.device

    # 1. 图上生成随机游走
    # 优化：使用 batch_generate_walks 替代 batch_random_walks_for_all_nodes
    # 避免在每个节点的循环中重复构建邻接表，显著提升速度
    adj_list = build_adj_list(edge_index, num_nodes)
    all_nodes_indices = list(range(num_nodes))
    
    # walks shape: [num_nodes, walks_per_node, walk_length + 1]
    walks = batch_generate_walks(adj_list, all_nodes_indices, walks_per_node, walk_length, device)
    
    # 2. 构建正样本对 (u, v)
    # 起点 u 是 walks 的第 0 列 -> 广播扩展到与 context 节点一样的形状
    u = walks[:, :, 0].unsqueeze(2).expand(-1, -1, walk_length) # [N, n, k]
    # v 是上下文节点 (context nodes)
    v_pos = walks[:, :, 1:] # [N, n, k]

    u = u.reshape(-1)
    v_pos = v_pos.reshape(-1)

    # 过滤掉自环或者游走失败留在原地的无效情况
    valid_mask = u != v_pos
    u = u[valid_mask]
    v_pos = v_pos[valid_mask]

    if u.size(0) == 0:
        return torch.tensor(0.0, device=device)

    # 获取正样本对的 embedding
    emb_u = student_node_emb[u]         # [num_pos, D]
    emb_v_pos = student_node_emb[v_pos] # [num_pos, D]

    # 3. 计算正样本 Loss: -log(sigmoid(u^T * v_pos))
    pos_logits = torch.sum(emb_u * emb_v_pos, dim=1)
    pos_loss = -torch.mean(F.logsigmoid(pos_logits))

    # 4. 全局负采样
    num_pos = u.size(0)
    # 对于图分类 batch，全局负采样在不同子图之间采样也是合理的，能增加不同图之间的特征对比区分度
    neg_idx = torch.randint(0, num_nodes, (num_pos, num_neg_samples), device=device)
    
    # 扩展 u 以便与 K 个负样本做矩阵乘法: [num_pos, 1, D]
    emb_u_exp = emb_u.unsqueeze(1)
    # 负样本 emb: [num_pos, K, D]
    emb_v_neg = student_node_emb[neg_idx]

    # 5. 计算负样本 Loss: -log(sigmoid(- u^T * v_neg))
    # bmm 计算内积，得到 [num_pos, K]
    neg_logits = torch.bmm(emb_v_neg, emb_u_exp.transpose(1, 2)).squeeze(2)
    neg_loss = -torch.mean(F.logsigmoid(-neg_logits))

    # 总的 DeepWalk 损失
    return pos_loss + neg_loss

def sale_mlp_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    student_node_emb: torch.Tensor,
    edge_index: torch.Tensor,
    # --- 权重 ---
    kl_weight: float = 1.0,
    sale_weight: float = 1.0,
    # --- 超参数 ---
    temperature: float = 1.0,
    walks_per_node: int = 2,
    walk_length: int = 3,
    num_neg_samples: int = 3
) -> dict:
    """
    SALE-MLP 蒸馏损失函数
    包含：
    1. KL 蒸馏损失 (Logits 对齐)
    2. SALE DeepWalk 损失 (结构对齐)
    """
    device = student_logits.device
    
    # 1. KL 蒸馏损失
    loss_kl = torch.tensor(0.0, device=device)
    if kl_weight > 0:
        loss_kl = kl_divergence_loss(student_logits, teacher_logits, temperature)
        
    # 2. SALE DeepWalk 损失
    loss_sale = torch.tensor(0.0, device=device)
    if sale_weight > 0:
        loss_sale = sale_deepwalk_loss(
            student_node_emb=student_node_emb,
            edge_index=edge_index,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            num_neg_samples=num_neg_samples
        )
        
    total_loss = kl_weight * loss_kl + sale_weight * loss_sale
    
    return {
        'total_loss': total_loss,
        'kl_loss': loss_kl,
        'sale_loss': loss_sale
    }