import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import math


class GraphTransformerBlock(nn.Module):
    """
    基于 TransformerConv 的图 Transformer 块（注意力 + 残差 + 前馈 + 归一化）

    结构：
    - 邻居多头注意力（TransformerConv）
    - 残差连接 + LayerNorm
    - 前馈网络（FFN）
    - 残差连接 + LayerNorm

    Args:
        in_dim (int): 输入特征维度
        out_dim (int): 输出（隐藏）特征维度
        heads (int): 多头注意力的头数
        dropout (float): Dropout 概率
        edge_dim (int | None): 边特征维度（如无边特征则设为 None）
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, edge_dim: int | None = None):
        super().__init__()
        # 注意力层（concat=False 让输出维度为 out_dim，而非 heads*out_dim）
        self.attn = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True,
        )

        # 当 in_dim != out_dim 时，为残差通路添加线性投影
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # 归一化与前馈网络
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, 2 * out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        # 注意力消息传递
        h = self.attn(x, edge_index, edge_attr)
        # 残差 + 归一化
        x = self.norm1(self.dropout(h) + self.res_proj(x))
        # 前馈网络
        h2 = self.ffn(x)
        # 残余 + 归一化
        x = self.norm2(self.dropout(h2) + x)
        return x

    def reset_parameters(self):
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)
            if self.res_proj.bias is not None:
                nn.init.zeros_(self.res_proj.bias)
        if hasattr(self.attn, 'reset_parameters'):
            self.attn.reset_parameters()
        # LayerNorm/Linear/Sequential 默认初始化即可


class MultiHeadGraphAttention(nn.Module):
    """
    原生多头注意力（基于边索引进行消息传递与聚合），不依赖 TransformerConv。

    输入：
    - x: [N, D] 节点特征
    - edge_index: [2, E] 边索引（从 cols -> rows 聚合信息）

    输出：
    - out: [N, D] 注意力聚合后的节点特征（同维度）
    - att: [E, H] 每条边、每个头的注意力权重（便于调试，训练中可忽略）
    """
    def __init__(self, hidden_dim: int, heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 heads={heads} 整除")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_head = hidden_dim // heads

        # Q/K/V 线性映射（用参数矩阵实现，保持与用户示例相近的风格）
        self.qTrans = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.kTrans = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.vTrans = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qTrans)
        nn.init.xavier_uniform_(self.kTrans)
        nn.init.xavier_uniform_(self.vTrans)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        device = x.device
        N, D = x.shape
        rows = edge_index[0].to(device)
        cols = edge_index[1].to(device)
        E = rows.size(0)
        H = self.heads
        Dh = self.dim_head

        # 线性投影到 Q/K/V
        q = x @ self.qTrans  # [N, D]
        k = x @ self.kTrans  # [N, D]
        v = x @ self.vTrans  # [N, D]

        # 按边采样出源/目标的嵌入
        q_e = q[rows, :]  # [E, D]
        k_e = k[cols, :]  # [E, D]
        v_e = v[cols, :]  # [E, D]

        # 分头视图
        q_e = q_e.view(E, H, Dh)
        k_e = k_e.view(E, H, Dh)
        v_e = v_e.view(E, H, Dh)

        # 注意力打分（缩放点积）
        att_logits = torch.einsum('ehd,ehd->eh', q_e, k_e) / math.sqrt(Dh)  # [E, H]
        att_logits = torch.clamp(att_logits, -10.0, 10.0)
        exp_att = torch.exp(att_logits)

        # 对每个节点（rows）和每个头做 softmax 归一化：sum_{邻居} exp_att
        denom = torch.zeros(N, H, device=device)
        denom.index_add_(0, rows, exp_att)
        att = exp_att / (denom[rows, :] + 1e-9)  # [E, H]
        att = self.attn_dropout(att)

        # 加权聚合 V（先按边计算，再按 rows 聚合到节点）
        out_e = torch.einsum('eh,ehd->ehd', att, v_e)  # [E, H, Dh]
        out_e = out_e.reshape(E, D)  # [E, D]
        out = torch.zeros(N, D, device=device)
        out.index_add_(0, rows, out_e)  # [N, D]

        return out, att


class GraphTransformerNativeBlock(nn.Module):
    """
    原生图 Transformer 块：MultiHeadGraphAttention + 残差 + 归一化 + FFN + 残差 + 归一化
    """
    def __init__(self, hidden_dim: int, heads: int = 8, dropout: float = 0.6, attn_dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadGraphAttention(hidden_dim=hidden_dim, heads=heads, attn_dropout=attn_dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, edge_index)
        x = self.norm1(x + self.dropout(h))
        h2 = self.ffn(x)
        x = self.norm2(x + self.dropout(h2))
        return x

    def reset_parameters(self):
        if hasattr(self.attn, 'reset_parameters'):
            self.attn.reset_parameters()
        # LayerNorm/Linear/Sequential 默认初始化即可


class GlobalMultiHeadAttention(nn.Module):
    """
    全局多头注意力机制 - 每个节点与图中所有其他节点计算注意力
    
    与局部图注意力不同，这里实现标准的全连接注意力：
    - 时间复杂度：O(N²)
    - 空间复杂度：O(N²)
    - 每个节点与所有节点计算注意力权重
    
    Args:
        hidden_dim (int): 隐藏维度
        heads (int): 注意力头数
        attn_dropout (float): 注意力dropout概率
    """
    def __init__(self, hidden_dim: int, heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 heads={heads} 整除")
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_head = hidden_dim // heads
        self.scale = math.sqrt(self.dim_head)
        
        # Q/K/V 线性变换
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, edge_bias: torch.Tensor | None = None, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [N, D] 节点特征
        Returns:
            out: [N, D] 注意力聚合后的节点特征
        """
        N, D = x.shape
        H = self.heads
        Dh = self.dim_head
        
        # 线性变换得到 Q, K, V
        q = self.q_proj(x)  # [N, D]
        k = self.k_proj(x)  # [N, D]
        v = self.v_proj(x)  # [N, D]
        
        # 重塑为多头形式
        q = q.view(N, H, Dh)  # [N, H, Dh]
        k = k.view(N, H, Dh)  # [N, H, Dh]
        v = v.view(N, H, Dh)  # [N, H, Dh]
        
        # 计算注意力分数 (全连接)
        # q @ k.T -> [N, H, Dh] @ [N, Dh, H] -> [N, H, N]
        att_scores = torch.einsum('nhd,mhd->nhm', q, k) / self.scale
        
        # 数值稳定性
        att_scores = torch.clamp(att_scores, -10.0, 10.0)
        if edge_index is not None and edge_bias is not None:
            rows = edge_index[0]
            cols = edge_index[1]
            att_scores[rows, :, cols] = att_scores[rows, :, cols] + edge_bias
        if batch is not None:
            mask = (batch.view(-1, 1) == batch.view(1, -1))
            mask = mask.unsqueeze(1).expand(-1, H, -1)
            att_scores = att_scores.masked_fill(~mask, -1e9)
        
        # Softmax归一化 (对每个查询节点的所有键节点)
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights_ = self.attn_dropout(att_weights)
        
        # 加权聚合值向量
        # att_weights @ v -> [N, H, N] @ [N, H, Dh] -> [N, H, Dh]
        out = torch.einsum('nhm,mhd->nhd', att_weights_, v)
        
        # 合并多头结果
        out = out.reshape(N, D)  # [N, D]
        
        # 输出投影
        out = self.out_proj(out)
        
        return out,att_weights


class GlobalGraphTransformerBlock(nn.Module):
    """
    全局图 Transformer 块：GlobalMultiHeadAttention + 残差 + 归一化 + FFN + 残差 + 归一化
    """
    def __init__(self, hidden_dim: int, heads: int = 8, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        self.attn = GlobalMultiHeadAttention(hidden_dim=hidden_dim, heads=heads, attn_dropout=attn_dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, edge_bias: torch.Tensor | None = None, batch: torch.Tensor | None = None) -> torch.Tensor:
        h,att_weights = self.attn(x, edge_index=edge_index, edge_bias=edge_bias, batch=batch)
        x = self.norm1(x + self.dropout(h))
        h2 = self.ffn(x)
        x = self.norm2(x + self.dropout(h2))
        return x,att_weights
    
    def reset_parameters(self):
        if hasattr(self.attn, 'reset_parameters'):
            self.attn.reset_parameters()


