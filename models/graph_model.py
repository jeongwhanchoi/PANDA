import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GATConv, GatedGraphConv, GINConv, FiLMConv, global_mean_pool
from torch_geometric.utils import degree
from models.layers import PandaGCNConv, PandaGINConv, PandaRGCNConv, PandaRGINConv
from torch_geometric.utils import to_networkx
import networkx as nx
import networkit as nk

class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        if self.layer_type in ["PANDA-GCN", "PANDA-GIN"]: 
            self.exp_factor = args.exp_factor
            self.in_features_exp = args.input_dim
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        self.top_k = args.top_k
        self.centrality_measure = args.centrality
        last_flag = False
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            if i == self.num_layers - 1:
                last_flag = True
            layers.append(self.get_layer(in_features, out_features, last_flag))
        self.layers = ModuleList(layers)
        print(self.layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            # add transformation associated with complete graph if last layer is fully adjacent
            if self.layer_type == "R-GCN" or self.layer_type == "GCN" or self.layer_type == "PANDA-GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN" or self.layer_type == "PANDA-GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(),nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError

    def get_layer(self, in_features, out_features, last_flag):          
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type == "PANDA-GCN":
            if last_flag:
                return PandaGCNConv(in_features, out_features, self.in_features_exp, out_features) 
            layer = PandaGCNConv(in_features, out_features, self.in_features_exp, int(out_features*self.exp_factor)) 
            self.in_features_exp = int(out_features*self.exp_factor)
            return layer
        elif self.layer_type == "PANDA-GIN":
            if last_flag:
                return PandaGINConv(in_features, out_features, self.in_features_exp, out_features) 
            layer = PandaGINConv(in_features, out_features, self.in_features_exp, int(out_features*self.exp_factor)) 
            self.in_features_exp = int(out_features*self.exp_factor)
            return layer
        elif self.layer_type == "PANDA-RGCN":
            if last_flag:
                return PandaRGCNConv(in_features, out_features, self.in_features_exp, out_features, self.num_relations) 
            layer = PandaRGCNConv(in_features, out_features, self.in_features_exp, int(out_features*self.exp_factor), self.num_relations) 
            self.in_features_exp = int(out_features*self.exp_factor)
            return layer
        elif self.layer_type == "PANDA-RGIN":
            if last_flag:
                return PandaRGINConv(in_features, out_features, self.in_features_exp, out_features, self.num_relations) 
            layer = PandaRGINConv(in_features, out_features, self.in_features_exp, int(out_features*self.exp_factor), self.num_relations) 
            self.in_features_exp = int(out_features*self.exp_factor)
            return layer

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()

        if self.layer_type in ["PANDA-GCN", "PANDA-GIN", "PANDA-RGCN", "PANDA-RGIN"]:
            if self.centrality_measure != 'degree_simple':
                centrality = graph.centrality
                exp_mask = torch.zeros(x.size(0), dtype=torch.bool)
                for i in range(len(ptr) - 1):
                    start, end = int(ptr[i]), int(ptr[i + 1])  # Graph node indices
                    centrality_values = centrality[start:end]
                    _, topk_indices = centrality_values.sort(descending=True)
                    topk_indices = topk_indices[:self.top_k]  # Select top-k

                    # Convert to global node indices
                    topk_global_indices = topk_indices + start
                    exp_mask[topk_global_indices] = True
            else:
                deg = degree(edge_index[0])
                topk_nodes_per_graph = []
                exp_mask = torch.zeros(x.size(0), dtype=torch.bool)
                for i in range(len(ptr) - 1):
                    start, end = int(ptr[i]), int(ptr[i + 1])
                    graph_degrees = deg[start:end] 
                    _, topk_indices = graph_degrees.sort(descending=True)
                    topk_indices = topk_indices[:self.top_k]

                    topk_global_indices = topk_indices + start
                    topk_nodes_per_graph.append(topk_global_indices)
                    exp_mask[topk_global_indices] = True

        for i, layer in enumerate(self.layers):
            if self.layer_type in ["PANDA-GCN", "PANDA-GIN"]:
                if i == self.num_layers - 1:
                    x_exp, x_nexp = layer(x_exp, x_nexp, edge_index, exp_mask)
                    x = torch.zeros((x_exp.shape[0] + x_nexp.shape[0], x_exp.shape[-1]), dtype=x_exp.dtype, device=x_exp.device)
                    x[~exp_mask] = x_nexp
                    x[exp_mask] = x_exp
                else:
                    if i == 0:
                        x_nexp = x[~exp_mask]
                        x_exp = x[exp_mask]
                        x_exp, x_nexp = layer(x_exp, x_nexp, edge_index, exp_mask)
                    else:
                        x_exp, x_nexp = layer(x_exp, x_nexp, edge_index, exp_mask)
                x_exp, x_nexp = self.act_fn(x_exp), self.act_fn(x_nexp)
                x_exp, x_nexp = self.dropout(x_exp), self.dropout(x_nexp)
            elif self.layer_type in ["PANDA-RGCN", "PANDA-RGIN"]:
                if i == self.num_layers - 1:
                    x_exp, x_nexp = layer(x_exp, x_nexp, edge_index, exp_mask, edge_type=graph.edge_type)
                    x = torch.zeros((x_exp.shape[0] + x_nexp.shape[0], x_exp.shape[-1]), dtype=x_exp.dtype, device=x_exp.device)
                    x[~exp_mask] = x_nexp
                    x[exp_mask] = x_exp
                else:
                    if i == 0:
                        x_nexp = x[~exp_mask]
                        x_exp = x[exp_mask]
                        x_exp, x_nexp = layer(x_exp, x_nexp, edge_index, exp_mask, edge_type=graph.edge_type)
                    else:
                        x_exp, x_nexp = layer(x_exp, x_nexp, edge_index, exp_mask, edge_type=graph.edge_type)
                x_exp, x_nexp = self.act_fn(x_exp), self.act_fn(x_nexp)
                x_exp, x_nexp = self.dropout(x_exp), self.dropout(x_nexp)
                
            else:
                if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                    x_new = layer(x, edge_index, edge_type=graph.edge_type)
                else:
                    x_new = layer(x, edge_index)
                if i != self.num_layers - 1:
                    x_new = self.act_fn(x_new)
                    x_new = self.dropout(x_new)
                if i == self.num_layers - 1 and self.args.last_layer_fa:
                    # handle final layer when making last layer FA
                    combined_values = global_mean_pool(x, batch)
                    combined_values = self.last_layer_transform(combined_values)
                    if self.layer_type in ["R-GCN", "R-GIN"]:
                        x_new += combined_values[batch]
                    else:
                        x_new = combined_values[batch]
                x = x_new 
        if measure_dirichlet:
            # check dirichlet energy instead of computing final values
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
        return x