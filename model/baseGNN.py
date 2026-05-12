from platform import node
import torch
from torch.fx import graph
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GINEConv, global_mean_pool, global_max_pool, global_add_pool, AttentionalAggregation 
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from layers.GNN import *
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class GIN(torch.nn.Module):
    def __init__(self, num_layers=4, input_dim=4, hidden_dim=32, num_classes=10, 
                dropout=0., pooling_method='attention', edge_attr=False, edge_attr_dim=None,
                useEmbedding=False, torchEmbedding=False, addVirtualNode=False,add_residual=False):
        super(GIN, self).__init__()
        self.num_features = input_dim
        self.edge_dim = edge_attr_dim
        hidden = hidden_dim
        
        self.useEmbedding = useEmbedding
        self.torchEmbedding = torchEmbedding
        self.addVirtualNode = addVirtualNode
        self.add_residual = add_residual
        # -------- Node / Edge Embedding ----------
        if self.useEmbedding:
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            #self.num_features = hidden
        elif self.torchEmbedding:
            # Input projection to hidden dim
            self.input_proj = nn.Embedding(self.num_features, hidden)
        else:
            # Input projection to hidden dim
            self.input_proj = nn.Linear(self.num_features, hidden)

        if edge_attr:
            if self.useEmbedding:
                self.bond_encoder = BondEncoder(hidden)
            elif self.torchEmbedding:
                self.edge_emb = nn.Embedding(self.edge_dim, hidden)
            else:
                self.edge_emb = nn.Linear(self.edge_dim, hidden)

        # -------- Virtual Node ----------
        if self.addVirtualNode:
            # 只有 1 个虚拟节点，对每个 graph 都复制
            self.virtual_node_emb = nn.Embedding(1, hidden)
            nn.init.constant_(self.virtual_node_emb.weight.data, 0)

            # 每层一个 virtual node MLP
            self.vn_mlps = nn.ModuleList()
            for i in range(num_layers):
                self.vn_mlps.append(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden)
                    )
                )

        # -------- Conv Layers ----------
        
        
        #first conv
        if edge_attr:
            self.conv1 = GINEConv(
                Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU()
                ),
                train_eps=True, edge_dim=hidden
            )
        else:
            self.conv1 = GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=True
            )

        # other layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            if edge_attr:
                self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True, edge_dim=hidden
                    )
                )
            else:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True
                    )
                )

        # ---------- Pooling + Prediction ----------
        self.lin1 = nn.Linear(num_layers * hidden, hidden)
       

        if self.add_residual:
            self.bn_final = nn.BatchNorm1d(hidden) 

        if pooling_method == 'attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
        elif pooling_method in ('sum', 'add'):
            self.pool = global_add_pool
        elif pooling_method == 'mean':
            self.pool = global_mean_pool
        elif pooling_method == 'max':
            self.pool = global_max_pool

        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(hidden, num_classes)

        self.apply(weight_reset)

    # ---------------------------------------------------------
    #                        FORWARD
    # ---------------------------------------------------------
    def forward(self, data, output_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #x = x[:, :self.num_features]

        if self.useEmbedding:
            # OGB: x是 [N, 9] -> Encoder -> [N, D]
            x = self.atom_encoder(x)

        elif self.torchEmbedding:
            if x.dtype == torch.float:
                x = x.long()
            if x.dim() == 2 and x.shape[1] == 1:
                x = x.squeeze(1)
            # Embedding Lookup
            x = self.input_proj(x)
        else:
            x = x.float()
            
            if x.shape[1] > self.num_features:
                x = x[:, :self.num_features]
            x = self.input_proj(x)

        if 'edge_attr' in data:
            e = data.edge_attr
            if self.useEmbedding:
                e = self.bond_encoder(e)
            elif self.torchEmbedding:
                if e.dtype == torch.float:
                    e = e.long()
                if e.dim() == 2 and e.shape[1] == 1:
                    e = e.squeeze(1)
                e = self.edge_emb(e)
            else:
                e = self.edge_emb(e)
        else:
            e = None

        # ----------- Virtual Node Init -----------
        if self.addVirtualNode:
            # 每个图一个虚拟节点特征
            virtualnode = self.virtual_node_emb.weight.repeat(batch.max().item()+1, 1)
        else:
            virtualnode = None

        # =======================
        #   Layer 1
        # =======================
        if self.addVirtualNode:
            x = x + virtualnode[batch]

        h = self.conv1(x, edge_index, e)

        if self.add_residual:
            x = x + h
        else:
            x = h
        xs = [x]

        # 更新虚拟节点
        if self.addVirtualNode:
            vn_update = global_add_pool(x, batch)          # 聚合整图节点
            virtualnode = virtualnode + self.vn_mlps[0](vn_update) 

        # =======================
        #   Other Layers
        # =======================
        for i, conv in enumerate(self.convs):
            if self.addVirtualNode:
                x = x + virtualnode[batch]

            h = conv(x, edge_index, e)

            if self.add_residual:
                x = x + h
            else:
                x = h
            xs.append(x)

            if self.addVirtualNode:
                vn_update = global_add_pool(x, batch)
                virtualnode = virtualnode + self.vn_mlps[i+1](vn_update)

        # =======================
        # Graph-level prediction
        # =======================
        h = F.relu(self.lin1(torch.cat(xs, dim=-1)))
        graph_h = self.pool(h, batch)
    
        
        if self.add_residual:
            graph_h = self.bn_final(graph_h)
            graph_h = F.relu(graph_h)

        end_h = self.dropout(graph_h)
        out = self.pred(end_h)

        if output_emb:
            return out, h, graph_h
        else:
            return out

class GIN_Drop(torch.nn.Module):
    def __init__(self, num_layers=4, input_dim=4, hidden_dim=32, num_classes=10, 
                dropout=0., pooling_method='attention', edge_attr=False, edge_attr_dim=None,
                useEmbedding=False, torchEmbedding=False, addVirtualNode=False,add_residual=False):
        super(GIN_Drop, self).__init__()
        self.num_features = input_dim
        self.edge_dim = edge_attr_dim
        hidden = hidden_dim
        
        self.useEmbedding = useEmbedding
        self.torchEmbedding = torchEmbedding
        self.addVirtualNode = addVirtualNode
        self.add_residual = add_residual
        # -------- Node / Edge Embedding ----------
        if self.useEmbedding:
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            #self.num_features = hidden
        elif self.torchEmbedding:
            # Input projection to hidden dim
            self.input_proj = nn.Embedding(self.num_features, hidden)
        else:
            # Input projection to hidden dim
            self.input_proj = nn.Linear(self.num_features, hidden)

        if edge_attr:
            if self.useEmbedding:
                self.bond_encoder = BondEncoder(hidden)
            elif self.torchEmbedding:
                self.edge_emb = nn.Embedding(self.edge_dim, hidden)
            else:
                self.edge_emb = nn.Linear(self.edge_dim, hidden)

        # -------- Virtual Node ----------
        if self.addVirtualNode:
            # 只有 1 个虚拟节点，对每个 graph 都复制
            self.virtual_node_emb = nn.Embedding(1, hidden)
            nn.init.constant_(self.virtual_node_emb.weight.data, 0)

            # 每层一个 virtual node MLP
            self.vn_mlps = nn.ModuleList()
            for i in range(num_layers):
                self.vn_mlps.append(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden)
                    )
                )

        # -------- Conv Layers ----------
        
        
        #first conv
        if edge_attr:
            self.conv1 = GINEConv(
                Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU()
                ),
                train_eps=True, edge_dim=hidden
            )
        else:
            self.conv1 = GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=True
            )

        # other layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            if edge_attr:
                self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True, edge_dim=hidden
                    )
                )
            else:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True
                    )
                )

        # ---------- Pooling + Prediction ----------
        self.lin1 = nn.Linear(num_layers * hidden, hidden)
       

        if self.add_residual:
            self.bn_final = nn.BatchNorm1d(hidden) 

        if pooling_method == 'attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
        elif pooling_method in ('sum', 'add'):
            self.pool = global_add_pool
        elif pooling_method == 'mean':
            self.pool = global_mean_pool
        elif pooling_method == 'max':
            self.pool = global_max_pool

        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(hidden, num_classes)

        self.apply(weight_reset)


    def forward(self, data, output_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 节点初始 Embedding
        if self.useEmbedding:
            x = self.atom_encoder(x)
        elif self.torchEmbedding:
            if x.dtype == torch.float: x = x.long()
            if x.dim() == 2 and x.shape[1] == 1: x = x.squeeze(1)
            x = self.input_proj(x)
        else:
            x = x.float()
            if x.shape[1] > self.num_features:
                x = x[:, :self.num_features]
            x = self.input_proj(x)

        # 2. 边初始 Embedding
        if 'edge_attr' in data and data.edge_attr is not None:
            e = data.edge_attr
            if self.useEmbedding:
                e = self.bond_encoder(e)
            elif self.torchEmbedding:
                if e.dtype == torch.float: e = e.long()
                if e.dim() == 2 and e.shape[1] == 1: e = e.squeeze(1)
                e = self.edge_emb(e)
            else:
                e = self.edge_emb(e)
        else:
            e = None

        # 3. 虚拟节点初始化
        if self.addVirtualNode:
            virtualnode = self.virtual_node_emb.weight.repeat(batch.max().item()+1, 1)
        else:
            virtualnode = None

        # =======================
        #   Layer 1
        # =======================
        if self.addVirtualNode:
            x = x + virtualnode[batch]

        h = self.conv1(x, edge_index, e)

        if self.add_residual:
            x = x + h
        else:
            x = h
            
        # 【关键修改 1】：第一层卷积后的 Dropout
        x = F.dropout(x, p=self.dropout.p, training=self.training)
        xs = [x]

        if self.addVirtualNode:
            vn_update = global_add_pool(x, batch)
            # 【关键修改 2】：虚拟节点更新后也加上 Dropout 防过拟合
            vn_update = F.dropout(vn_update, p=self.dropout.p, training=self.training)
            virtualnode = virtualnode + self.vn_mlps[0](vn_update) 

        # =======================
        #   Other Layers
        # =======================
        for i, conv in enumerate(self.convs):
            if self.addVirtualNode:
                x = x + virtualnode[batch]

            h = conv(x, edge_index, e)

            if self.add_residual:
                x = x + h
            else:
                x = h
                
            # 【关键修改 3】：后续卷积层的 Dropout
            x = F.dropout(x, p=self.dropout.p, training=self.training)
            xs.append(x)

            if self.addVirtualNode:
                vn_update = global_add_pool(x, batch)
                vn_update = F.dropout(vn_update, p=self.dropout.p, training=self.training)
                virtualnode = virtualnode + self.vn_mlps[i+1](vn_update)

        # =======================
        # Graph-level prediction
        # =======================
        # 把所有层的节点特征 concat 起来 (Jumping Knowledge)
        h = F.relu(self.lin1(torch.cat(xs, dim=-1)))
        graph_h = self.pool(h, batch)
        
        if self.add_residual:
            graph_h = self.bn_final(graph_h)
            graph_h = F.relu(graph_h)

        # 最终分类前的 Dropout 保留
        end_h = self.dropout(graph_h)
        out = self.pred(end_h)

        if output_emb:
            return out, h, graph_h
        else:
            return out


class GCN(torch.nn.Module):
    def __init__(self, num_layers=4, input_dim=4, hidden_dim=32, num_classes=10, 
                dropout=0., pooling_method='attention', edge_attr=False, edge_attr_dim=None,
                useEmbedding=False, torchEmbedding=False, addVirtualNode=False, add_residual=False):
        super(GCN, self).__init__()
        self.num_features = input_dim
        self.edge_dim = edge_attr_dim
        hidden = hidden_dim
        
        self.useEmbedding = useEmbedding
        self.torchEmbedding = torchEmbedding
        self.addVirtualNode = addVirtualNode
        self.add_residual = add_residual
        self.edge_attr = edge_attr
        
        # -------- Node / Edge Embedding ----------
        if self.useEmbedding:
            from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
        elif self.torchEmbedding:
            self.input_proj = nn.Embedding(self.num_features, hidden)
        else:
            self.input_proj = nn.Linear(self.num_features, hidden)

        if edge_attr:
            if self.useEmbedding:
                pass # bond_encoder acts as edge embedding
            elif self.torchEmbedding:
                self.edge_emb = nn.Embedding(self.edge_dim if self.edge_dim is not None else 10, hidden)
            else:
                self.edge_emb = nn.Linear(self.edge_dim if self.edge_dim is not None else 1, hidden)

        # -------- Virtual Node ----------
        if self.addVirtualNode:
            self.virtual_node_emb = nn.Embedding(1, hidden)
            nn.init.constant_(self.virtual_node_emb.weight.data, 0)

            self.vn_mlps = nn.ModuleList()
            for i in range(num_layers):
                self.vn_mlps.append(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden)
                    )
                )

        # -------- Convolutions ----------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # GCNConv inherently doesn't handle multidimensional edge features in the same way GINE does.
        # But we can pass edge weights or process edge features if needed. 
        # For simplicity and structural consistency with your GIN, we build standard GCN layers.
        self.conv1 = GCNConv(hidden, hidden)
        self.bns.append(nn.BatchNorm1d(hidden))
        
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        # -------- Pooling ----------
        if pooling_method == "sum" or pooling_method == "add":
            self.pool = global_add_pool
        elif pooling_method == "mean":
            self.pool = global_mean_pool
        elif pooling_method == "max":
            self.pool = global_max_pool
        elif pooling_method == "attention":
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
                nn.Linear(hidden * num_layers, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            ))
        else:
            raise ValueError(f"Pooling {pooling_method} not supported")

        # -------- Prediction ----------
        self.lin1 = nn.Linear(hidden * num_layers, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data, output_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Node feature embedding
        if self.useEmbedding:
            x = self.atom_encoder(x)
        elif self.torchEmbedding:
            if x.dtype == torch.float: x = x.long()
            if x.dim() == 2 and x.shape[1] == 1: x = x.squeeze(1)
            x = self.input_proj(x)
        else:
            x = x.float()
            if x.shape[1] > self.num_features:
                x = x[:, :self.num_features]
            x = self.input_proj(x)

        # 2. Edge feature embedding (Optional for GCN, but computed for consistency)
        e = None
        if self.edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            e_feat = data.edge_attr
            if self.useEmbedding:
                e = self.bond_encoder(e_feat)
            elif self.torchEmbedding:
                if e_feat.dtype == torch.float: e_feat = e_feat.long()
                if e_feat.dim() == 2 and e_feat.shape[1] == 1: e_feat = e_feat.squeeze(1)
                e = self.edge_emb(e_feat)
            else:
                e = self.edge_emb(e_feat.float())
                
        # GCN usually takes 1D edge_weight. If we have edge features 'e', 
        # we can optionally map them to weights, but standard PyG GCN expects a scalar weight.
        # To maintain strict compatibility without breaking PyG, we omit 'e' in forward if it's multidimensional,
        # unless you use a custom GCN that supports it. Here we use standard GCNConv (ignores multidimensional 'e').

        # ----------- Virtual Node Init -----------
        if self.addVirtualNode:
            virtualnode = self.virtual_node_emb.weight.repeat(batch.max().item()+1, 1)
        else:
            virtualnode = None

        # =======================
        #   Layer 1
        # =======================
        if self.addVirtualNode:
            x = x + virtualnode[batch]

        # First convolution
        # Note: GCNConv takes edge_weight, not edge_attr. We pass edge_index.
        h = self.conv1(x, edge_index)
        h = self.bns[0](h)
        h = F.relu(h)

        if self.add_residual:
            x = x + h
        else:
            x = h
        xs = [x]

        if self.addVirtualNode:
            vn_update = global_add_pool(x, batch)
            virtualnode = virtualnode + self.vn_mlps[0](vn_update) 

        # =======================
        #   Other Layers
        # =======================
        for i, conv in enumerate(self.convs):
            if self.addVirtualNode:
                x = x + virtualnode[batch]

            h = conv(x, edge_index)
            h = self.bns[i+1](h)
            h = F.relu(h)

            if self.add_residual:
                x = x + h
            else:
                x = h
            xs.append(x)

            if self.addVirtualNode:
                vn_update = global_add_pool(x, batch)
                virtualnode = virtualnode + self.vn_mlps[i+1](vn_update)

        # =======================
        # Graph-level prediction
        # =======================
        node_h = torch.cat(xs, dim=-1)
        h = F.relu(self.lin1(node_h))
        graph_h = self.pool(h, batch)
        
        graph_h_drop = F.dropout(graph_h, p=self.dropout, training=self.training)
        logits = self.lin2(graph_h_drop)

        if output_emb:
            return logits, h, graph_h
        return logits
    



class SAGE(torch.nn.Module):
    def __init__(self, num_layers=4, input_dim=4, hidden_dim=32, num_classes=10, 
                dropout=0., pooling_method='attention', edge_attr=False, edge_attr_dim=None,
                useEmbedding=False, torchEmbedding=False, addVirtualNode=False, add_residual=False):
        super(SAGE, self).__init__()
        self.num_features = input_dim
        self.edge_dim = edge_attr_dim
        hidden = hidden_dim
        
        self.useEmbedding = useEmbedding
        self.torchEmbedding = torchEmbedding
        self.addVirtualNode = addVirtualNode
        self.add_residual = add_residual
        self.edge_attr = edge_attr
        
        # -------- Node / Edge Embedding ----------
        if self.useEmbedding:
            from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
        elif self.torchEmbedding:
            self.input_proj = nn.Embedding(self.num_features, hidden)
        else:
            self.input_proj = nn.Linear(self.num_features, hidden)

        if edge_attr:
            if self.useEmbedding:
                pass 
            elif self.torchEmbedding:
                self.edge_emb = nn.Embedding(self.edge_dim if self.edge_dim is not None else 10, hidden)
            else:
                self.edge_emb = nn.Linear(self.edge_dim if self.edge_dim is not None else 1, hidden)

        # -------- Virtual Node ----------
        if self.addVirtualNode:
            self.virtual_node_emb = nn.Embedding(1, hidden)
            nn.init.constant_(self.virtual_node_emb.weight.data, 0)

            self.vn_mlps = nn.ModuleList()
            for i in range(num_layers):
                self.vn_mlps.append(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden)
                    )
                )

        # -------- Convolutions ----------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # SAGEConv uses mean aggregation by default
        self.conv1 = SAGEConv(hidden, hidden)
        self.bns.append(nn.BatchNorm1d(hidden))
        
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        # -------- Pooling ----------
        if pooling_method == "sum" or pooling_method == "add":
            self.pool = global_add_pool
        elif pooling_method == "mean":
            self.pool = global_mean_pool
        elif pooling_method == "max":
            self.pool = global_max_pool
        elif pooling_method == "attention":
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
                nn.Linear(hidden * num_layers, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            ))
        else:
            raise ValueError(f"Pooling {pooling_method} not supported")

        # -------- Prediction ----------
        self.lin1 = nn.Linear(hidden * num_layers, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data, output_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Node feature embedding
        if self.useEmbedding:
            x = self.atom_encoder(x)
        elif self.torchEmbedding:
            if x.dtype == torch.float: x = x.long()
            if x.dim() == 2 and x.shape[1] == 1: x = x.squeeze(1)
            x = self.input_proj(x)
        else:
            x = x.float()
            if x.shape[1] > self.num_features:
                x = x[:, :self.num_features]
            x = self.input_proj(x)

        # ----------- Virtual Node Init -----------
        if self.addVirtualNode:
            virtualnode = self.virtual_node_emb.weight.repeat(batch.max().item()+1, 1)
        else:
            virtualnode = None

        # =======================
        #   Layer 1
        # =======================
        if self.addVirtualNode:
            x = x + virtualnode[batch]

        # SAGE Convolution
        h = self.conv1(x, edge_index)
        h = self.bns[0](h)
        h = F.relu(h)

        if self.add_residual:
            x = x + h
        else:
            x = h
        xs = [x]

        if self.addVirtualNode:
            vn_update = global_add_pool(x, batch)
            virtualnode = virtualnode + self.vn_mlps[0](vn_update) 

        # =======================
        #   Other Layers
        # =======================
        for i, conv in enumerate(self.convs):
            if self.addVirtualNode:
                x = x + virtualnode[batch]

            h = conv(x, edge_index)
            h = self.bns[i+1](h)
            h = F.relu(h)

            if self.add_residual:
                x = x + h
            else:
                x = h
            xs.append(x)

            if self.addVirtualNode:
                vn_update = global_add_pool(x, batch)
                virtualnode = virtualnode + self.vn_mlps[i+1](vn_update)

        # =======================
        # Graph-level prediction
        # =======================
        node_h = torch.cat(xs, dim=-1)
        h = F.relu(self.lin1(node_h))
        graph_h = self.pool(h, batch)
        
        graph_h_drop = F.dropout(graph_h, p=self.dropout, training=self.training)
        logits = self.lin2(graph_h_drop)

        if output_emb:
            return logits, h, graph_h
        return logits