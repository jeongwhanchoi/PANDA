import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool
from models.layers import PandaGCNConv, PandaGINConv
from torch_geometric.utils import to_networkx
import networkx as nx
import networkit as nk
from torch_geometric.utils import degree

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

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
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

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features, last_flag):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
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
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        if self.layer_type in ["PANDA-GCN", "PANDA-GIN"]:
            # Select the centrality measure
            centrality_measure = self.centrality_measure

            if centrality_measure != 'degree_simple':
                # Convert PyG graph to NetworkX graph for centrality measures
                G = to_networkx(graph, to_undirected=True)
                # Placeholder for centrality computation
                centrality = None
                exp_mask = torch.zeros(x.size(0), dtype=torch.bool)

                # Compute centrality based on the selected measure
                if centrality_measure == 'betweenness':
                    centrality = nx.betweenness_centrality(G)
                elif centrality_measure == 'eigen':
                    centrality = nx.eigenvector_centrality(G, max_iter=1000)
                elif centrality_measure == 'closeness':
                    centrality = nx.closeness_centrality(G)
                elif centrality_measure == 'degree':
                    centrality = nx.degree_centrality(G)
                elif centrality_measure == 'pagerank':
                    centrality = nx.pagerank(G)
                elif centrality_measure == 'katz':
                    centrality = nx.katz_centrality(G)
                elif centrality_measure == 'laplacian':
                    centrality = nx.laplacian_centrality(G)
                elif centrality_measure == 'second':
                    centrality = nx.second_order_centrality(G)
                elif centrality_measure == 'harmonic':
                    centrality = nx.harmonic_centrality(G)
                elif centrality_measure == 'load':
                    centrality = nx.load_centrality(G)
                elif centrality_measure == 'current':
                    centrality = nx.current_flow_betweenness_centrality(G)
                elif centrality_measure == 'betweenness-nk':
                    G = nk.nxadapter.nx2nk(G)
                    centrality = nk.centrality.Betweenness(G)
                    centrality.run()
                    centrality = centrality.scores()
                elif centrality_measure == 'approx_betweenness-nk':
                    G = nk.nxadapter.nx2nk(G)
                    centrality = nk.centrality.ApproxBetweenness(G, epsilon=0.1)
                    centrality.run()
                    centrality = centrality.scores()
                elif centrality_measure == 'closeness-nk':
                    G = nk.nxadapter.nx2nk(G)
                    centrality = nk.centrality.Closeness(G)
                    centrality.run()
                    centrality = centrality.scores()
                elif centrality_measure == 'approx_closeness-nk':
                    G = nk.nxadapter.nx2nk(G)
                    centrality = nk.centrality.ApproxCloseness(G, nSamples=10, epsilon=0.1)
                    centrality.run()
                    centrality = centrality.scores()
                else:
                    raise NameError('Centrality measure not found')

                # Convert centrality to tensor and sort
                if centrality_measure in ['betweenness-nk', 'approx_betweenness-nk', 'closeness-nk', 'approx_closeness-nk', 'eigen-nk']:
                    centrality_values = torch.tensor(centrality)
                else:
                    centrality_values = torch.tensor(list(centrality.values()))
                
                _, topk_indices = centrality_values.sort(descending=True)
                topk_indices = topk_indices[:self.top_k]  # Select top-k

                exp_mask[topk_indices] = True
            else:
                deg = degree(edge_index[0])
                exp_mask = torch.zeros(x.size(0), dtype=torch.bool)
                _, topk_indices = deg.sort(descending=True)
                topk_indices = topk_indices[:self.top_k]
                exp_mask[topk_indices] = True

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
            else:
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x = layer(x, edge_index, edge_type=graph.edge_type)
                else:
                    x = layer(x, edge_index)
                if i != self.num_layers - 1:
                    x = self.act_fn(x)
                    x = self.dropout(x)
        return x
