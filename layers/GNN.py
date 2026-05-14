import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing, GINEConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, softmax




class EdgeGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr='add', 
                useNodes = False):
        super().__init__(aggr=aggr)  # 'add' aggregation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_lin = nn.Linear(edge_dim, out_channels)
        self.useNodes = useNodes
        # 将 edge_attr 映射成一个标量权重（也可以是向量）        
        if self.useNodes:
            self.edge_update_mlp = update_edge_withNodes(3*out_channels, out_channels)
            self.update_lin = nn.Linear(out_channels, 1)

        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.GELU(),
                nn.Linear(out_channels, 1)   # 输出 scalar weight
            )

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_channels)
        # edge_attr: (E, edge_dim)
        x = self.lin(x)  # (N, out_channels)
        
        # add self loops to maintain GCN semantics
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr,
                                              num_nodes=x.size(0),
                                              fill_value=0.0)  # self-loop attr = 0
        edge_attr = self.edge_lin(edge_attr) # (E, out_channels)
        # compute normalization like GCN
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)  # out-degree (since message goes i <- j)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # compute edge scalar weights
        if self.useNodes:
            edge_attr = self.edge_update_mlp(x, edge_index, edge_attr)
            edge_attr = F.gelu(edge_attr)
            edge_weight = self.update_lin(edge_attr).squeeze(-1)   # (E,)
        else:
            edge_weight = self.edge_mlp(edge_attr).squeeze(-1)   # (E,)
        
        # norm factor per edge = deg^-1/2(row) * deg^-1/2(col)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # (E,)

        # propagate: message will be x_j * edge_weight * norm
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, norm=norm) + self.bias

    def message(self, x_j, edge_weight, norm):
        # x_j: (E, out_channels)
        # edge_weight: (E,)
        return (edge_weight.view(-1,1) * x_j) * norm.view(-1,1)

    def update(self, aggr_out):
        return aggr_out
        

class EdgeGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=1, concat=True, negative_slope=0.2, dropout=0.0, useNodes=False):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.useNodes = useNodes

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        # edge embedding to same dim as per-head out_dim (or compress)
        self.edge_lin = nn.Linear(edge_dim, heads * out_channels, bias=False)

        # attention: a vector per head
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels * 2))  # for [Wh_i || Wh_j] part
        self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))  # for edge contribution

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self.useNodes:
            self.edge_update_mlp = nn.Linear(3 * out_channels, out_channels)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.att_edge)
        if hasattr(self, 'edge_update_mlp'):
            nn.init.xavier_uniform_(self.edge_update_mlp.weight)
            if self.edge_update_mlp.bias is not None:
                nn.init.zeros_(self.edge_update_mlp.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_channels)
        H = self.lin(x)  # (N, heads*out_channels)
        H = H.view(-1, self.heads, self.out_channels)  # (N, heads, out_dim)
        E = self.edge_lin(edge_attr)  # (E, heads*out_dim)
        E = E.view(-1, self.heads, self.out_channels)  # (E, heads, out_dim)

        return self.propagate(edge_index, x=H, edge_attr=E)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # x_i, x_j: (E, heads, out_dim)
        # edge_attr: (E, heads, out_dim)
        # compute attention score per head:
        # score = a^T [x_i || x_j] + a_edge^T edge_attr (then LeakyReLU)
        if self.useNodes:
            u = torch.cat([x_i, x_j, edge_attr], dim=-1)
            u2 = self.edge_update_mlp(u.reshape(-1, 3 * self.out_channels))
            u2 = u2.reshape(-1, self.heads, self.out_channels)
            edge_attr = torch.nn.functional.gelu(edge_attr + u2)
        cat = torch.cat([x_i, x_j], dim=-1)
        att_score = (cat * self.att).sum(dim=-1)
        edge_score = (edge_attr * self.att_edge).sum(dim=-1)
        score = self.leaky_relu(att_score + edge_score)
        alpha = softmax(score, index)
        alpha = self.dropout(alpha).unsqueeze(-1)  # (E, heads, 1)

        # message = alpha * x_j
        return x_j * alpha

    def update(self, aggr_out):
        # aggr_out: (N, heads, out_dim)
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)  # concat heads
        else:
            aggr_out = aggr_out.mean(dim=1)  # average heads
        return aggr_out


class update_edge_withNodes(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(update_edge_withNodes,self).__init__()
        self.lin = nn.Linear(input_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        rows, cols = edge_index
        x_rows = x[rows]
        x_cols = x[cols]
        edge_u = self.lin(torch.cat([x_rows, x_cols, edge_attr], dim=-1))
        edge_attr = F.gelu(edge_attr + edge_u)
        return edge_attr
        
