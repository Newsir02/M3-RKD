import torch.nn.functional as F
import math
from torch import nn
import torch
from torch.nn import functional as F
from layers.transformers import GlobalGraphTransformerBlock
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool,AttentionalAggregation
from torch_geometric.utils import add_self_loops,to_dense_adj
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import to_dense_batch




class StudentMLP_with_PE(nn.Module):
    def __init__(self, input_dim, pe_dim, hidden_dim, output_dim, num_layers=3, 
                 use_embedding=False, dropout=0.5, pool_type='add',add_pe=True):
        super().__init__()
        
        self.use_embedding = use_embedding
        self.pe_dim = pe_dim
        self.add_pe = add_pe
        # 1. 输入编码
        if self.use_embedding:
            self.node_emb = AtomEncoder(hidden_dim)
        else:
            self.node_emb = nn.Linear(input_dim, hidden_dim)

        # 2. 位置编码 (PE)
        if pe_dim > 0 and self.add_pe:
            self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
            self.pe_norm = nn.LayerNorm(hidden_dim)

        # 3. 纯 MLP 层 (不接受 edge_index)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                BN(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(num_layers-2)  #去掉 输入和输出 的 linear
        ])
        
        # 4. 聚合与分类
        #self.lin1 = nn.Linear(num_layers * hidden_dim, hidden_dim)
        
        if pool_type == 'add':
            self.pool = global_add_pool
        elif pool_type == 'mean':
            self.pool = global_mean_pool

            
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, output_emd=False, return_attn=False): 
        x, batch = data.x, data.batch
        
        # 1. 节点嵌入 + PE注入 (保持你原有的逻辑)
        if self.use_embedding:
            x = self.node_emb(x)
        else:
            x = self.node_emb(x.float())

        if self.pe_dim > 0 and self.add_pe and hasattr(data, 'laplacian_pe'):
            #print("Using Laplacian PE in MLP")
            pe = data.laplacian_pe
            if self.training:
                sign_flip = torch.rand(1, self.pe_dim, device=pe.device)
                sign_flip = torch.where(sign_flip > 0.5, 1.0, -1.0)
                pe = pe * sign_flip
            pe = self.pe_encoder(pe)
            pe = self.pe_norm(pe)
            x = x + pe

        # 2. MLP 处理
        #xs = []
        for layer in self.layers:
            x = layer(x)
            #xs.append(x)
        
        # 获得最终节点特征
        #node_emb = F.relu(self.lin1(torch.cat(xs, dim=-1)))
        node_emb = x
        # 3. 【关键修改】计算稀疏注意力 (仅针对 spatial_pe_idx 中的边)
        attn_values = None
        if return_attn and hasattr(data, 'spatial_pe_idx'):
            src, dst = data.spatial_pe_idx
            q = node_emb[src] 
            k = node_emb[dst]
            
            scale = 1.0 / math.sqrt(node_emb.size(-1))
            score = (q * k).sum(dim=-1) * scale
            attn_values = score
        # 4. 池化与分类
        graph_emb = self.pool(node_emb, batch)
        graph_emb = self.dropout(graph_emb)
        out = self.classifier(graph_emb)
        
        if return_attn:
            # 直接返回这个一维向量，不要放入 list (除非为了兼容性)
            return out, node_emb, graph_emb, [attn_values]
        elif output_emd:
            return out, node_emb, graph_emb
        else:
            return out
        
        
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_layers=3,
        dropout=0.5,
        pooling='add',
        useEmbedding=False,      # OGB 专用
        torchEmbedding=False,   # 【新增】ZINC 专用 (标准 nn.Embedding)
        pe_dim=0,
        add_pe=False,
        add_residual=False,
    ):
        super().__init__()

        self.useEmbedding = useEmbedding
        self.torchEmbedding = torchEmbedding
        self.dropout = dropout
        self.pe_dim = pe_dim
        self.add_pe = add_pe

        if useEmbedding:
            self.atom_encoder = AtomEncoder(hidden_dim)
            node_dim = hidden_dim
        elif torchEmbedding:
            self.input_proj = nn.Embedding(input_dim, hidden_dim)
            node_dim = hidden_dim
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            node_dim = hidden_dim

        if pe_dim > 0 and self.add_pe:
            self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
            self.pe_norm = nn.LayerNorm(hidden_dim)

        # -------- 3. Node-wise MLP --------
        self.node_mlps = nn.ModuleList()
        for i in range(num_layers - 2):
            self.node_mlps.append(
                nn.Sequential(
                    nn.Linear(node_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
            )
            node_dim = hidden_dim

        self.Dropout = nn.Dropout(self.dropout)

        if pooling in ('add', 'sum'):
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.add_residual = add_residual

        
        self.graph_bn = nn.BatchNorm1d(hidden_dim) 

        self.graph_mlp = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, output_emb=False, return_attn=False):
        x, batch = data.x, data.batch

        if self.useEmbedding:
            x = self.atom_encoder(x)
        
        elif self.torchEmbedding:
            if x.dtype == torch.float:
                x = x.long()
            if x.dim() == 2 and x.shape[1] == 1:
                x = x.squeeze(1)
            x = self.input_proj(x)
            
        else:
            x = self.input_proj(x.float())
        if self.pe_dim > 0 and self.add_pe and hasattr(data, 'laplacian_pe'):
            pe = data.laplacian_pe
            if self.training:
                sign_flip = torch.rand(1, self.pe_dim, device=pe.device)
                sign_flip = torch.where(sign_flip > 0.5, 1.0, -1.0)
                pe = pe * sign_flip
                
            pe_encoded = self.pe_encoder(pe)
            pe_encoded = self.pe_norm(pe_encoded)
            # 将 PE 加到节点特征上
            x = x + pe_encoded
        
        for mlp in self.node_mlps:
            node_emb = mlp(x)
            if self.add_residual:
                out = self.Dropout(node_emb)
                x = x + out
                node_emb = x
            else:
                x = self.Dropout(node_emb)

        graph_emb = self.pool(node_emb, batch)
        h_graph_emb = self.graph_bn(graph_emb)
        h_graph_emb = F.relu(h_graph_emb)
            
        h_emb = self.Dropout(h_graph_emb)
        out = self.graph_mlp(h_emb)
        if return_attn:
            return out, node_emb, graph_emb, None
        if output_emb:
            return out, node_emb, graph_emb
        else:
            return out


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         hidden_dim,
#         output_dim,
#         num_layers=3,
#         dropout=0.5,
#         pooling='mean',
#         norm_type='batch',  # GLNN 风格：支持 'batch', 'layer', 'none'
#         pe_dim=0,
#         add_pe=False,
#         **kwargs # 忽略多余参数
#     ):
#         super(MLP, self).__init__()
        
#         self.num_layers = num_layers
#         self.dropout_ratio = dropout
#         self.norm_type = norm_type
#         self.pooling = pooling
#         self.add_pe = add_pe
#         self.pe_dim = pe_dim
#         self.input_dim = input_dim
#         # -------- 1. Input Projection (替换 AtomEncoder) --------
#         # 直接使用 Linear 将输入特征映射到隐藏层维度
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
        
#         # 输入层的 Norm (GLNN 通常在每层 Linear 后都加 Norm)
#         if self.norm_type == "batch":
#             self.input_norm = nn.BatchNorm1d(hidden_dim)
#         elif self.norm_type == "layer":
#             self.input_norm = nn.LayerNorm(hidden_dim)
#         else:
#             self.input_norm = nn.Identity()

#         # -------- 2. Positional Encoding --------
#         if pe_dim > 0 and self.add_pe:
#             self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
#             # PE 也通常加一个 Norm
#             self.pe_norm = nn.LayerNorm(hidden_dim)

#         # -------- 3. Hidden Layers (GLNN Style) --------
#         # 使用 ModuleList 管理中间层，不使用 Sequential，方便控制流程
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()

#         # 构建中间层 (num_layers - 2 个，因为减去了输入层和输出层)
#         for _ in range(num_layers - 2):
#             self.layers.append(nn.Linear(hidden_dim, hidden_dim))
#             if self.norm_type == "batch":
#                 self.norms.append(nn.BatchNorm1d(hidden_dim))
#             elif self.norm_type == "layer":
#                 self.norms.append(nn.LayerNorm(hidden_dim))
#             else:
#                 self.norms.append(nn.Identity())

#         # -------- 4. Pooling Function --------
#         if pooling == 'mean':
#             self.pool = global_mean_pool
#         elif pooling == 'add' or pooling == 'sum':
#             self.pool = global_add_pool
#         elif pooling == 'max':
#             self.pool = global_max_pool
#         else:
#             raise ValueError(f"Unknown pooling type: {pooling}")

#         # -------- 5. Output Head (Classifier) --------
#         # 池化后的最终分类层
#         self.classifier = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, data, output_emb=False, return_attn=False):
#         x, batch = data.x, data.batch
#         #x = x[:,:self.input_dim]
#         # 1. Input Linear Projection
#         # 关键：不使用 AtomEncoder，直接 Linear。
#         # OGB 的 x 通常是 Long (int)，必须转为 Float 才能进 Linear。
#         x = x.float() 
#         x = self.input_proj(x)
        
#         # Norm -> ReLU -> Dropout (GLNN 标准 Block)
#         if self.norm_type != "none":
#             x = self.input_norm(x)
#         x = F.relu(x)
#         node_emb = x 
#         x = self.dropout(x)

#         # 2. Add Positional Encoding (如果开启)
#         if self.pe_dim > 0 and self.add_pe and hasattr(data, 'laplacian_pe'):
#             pe = data.laplacian_pe
#             if self.training:
#                 # 随机翻转符号
#                 sign_flip = torch.rand(1, self.pe_dim, device=pe.device)
#                 sign_flip = torch.where(sign_flip > 0.5, 1.0, -1.0)
#                 pe = pe * sign_flip
            
#             pe_encoded = self.pe_encoder(pe)
#             pe_encoded = self.pe_norm(pe_encoded)
#             # 将 PE 加到节点特征上
#             x = x + pe_encoded

#         # 3. Hidden Layers Forward
#         # 纯粹的堆叠，无残差连接
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if self.norm_type != "none":
#                 x = self.norms[i](x)
#             x = F.relu(x)
#             node_emb=x
#             x = self.dropout(x)
        
#         # 此时 x 是节点级嵌入 (Node Embeddings)
#         # h = x

#         # 4. Pooling
#         graph_emb = self.pool(node_emb, batch)

#         # 5. Final Prediction
#         # 通常分类头前不加 Norm，但可以加 Dropout
#         out = self.classifier(self.dropout(graph_emb)) # 可选双重 Dropout
#         # out = self.classifier(graph_emb)

#         # 6. Return for Distillation
#         if return_attn:
#             return out, node_emb,graph_emb, None
#         if output_emb:
#             return out, node_emb, graph_emb
#         else:
#             return out