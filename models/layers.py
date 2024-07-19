from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
import torch
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
import torch.nn.functional as F

class PandaGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_channels_exp, out_channels_exp, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_max_degree = out_channels_exp
        self.exp_dim = out_channels_exp
        self.nexp_dim = out_channels
        self.improved = False
        self.add_self_loops = True

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_max_degree = torch.nn.Parameter(torch.Tensor(in_channels_exp, out_channels_exp))
        self.select = torch.nn.Linear(out_channels+out_channels_exp, out_channels_exp)
        self.lin_ne_expansion = torch.nn.Linear(out_channels, out_channels_exp)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight_max_degree)
        torch.nn.init.xavier_uniform_(self.select.weight)
        torch.nn.init.xavier_uniform_(self.lin_ne_expansion.weight)

    def forward(self, x_exp, x_nexp, edge_index, max_degree_mask, edge_weight=None):
        # Apply weights
        x_normal =  x_nexp @ self.weight
        x_max_degree = x_exp @ self.weight_max_degree
        
        node_dim = x_normal.shape[0]+x_max_degree.shape[0]


        padding_size = x_max_degree.shape[-1] - x_normal.shape[-1]
        padding = torch.zeros(x_normal.size(0), padding_size, device=x_normal.device)
        x_normal_padded = torch.cat([x_normal, padding], dim=-1)

        x_combined = torch.zeros((node_dim, x_max_degree.shape[-1]), dtype=x_normal.dtype, device=x_normal.device)
        # # Place the results back according to the original node order
        x_combined[~max_degree_mask] = x_normal_padded
        x_combined[max_degree_mask] = x_max_degree
        
        # Normalize and propagate
        if isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight)
        else:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, node_dim, self.improved, self.add_self_loops)

        top_k_nodes = max_degree_mask.nonzero().to(x_normal.device)
        mask_en = torch.isin(edge_index[0], top_k_nodes) & ~torch.isin(edge_index[1], top_k_nodes)
        mask_ne = torch.isin(edge_index[1], top_k_nodes) & ~torch.isin(edge_index[0], top_k_nodes)
        mask_nn = ~torch.any(torch.isin(edge_index, top_k_nodes), dim=0)
        mask_ee = torch.isin(edge_index[0], top_k_nodes) & torch.isin(edge_index[1], top_k_nodes)

        edge_index_nn = edge_index[:, mask_nn]
        edge_index_ee = edge_index[:, mask_ee]
        edge_index_en = edge_index[:, mask_en]
        edge_index_ne = edge_index[:, mask_ne]
        edge_weight_nn = edge_weight[mask_nn]
        edge_weight_ee = edge_weight[mask_ee]
        edge_weight_en = edge_weight[mask_en]
        edge_weight_ne = edge_weight[mask_ne]

        out_nn = self.propagate(edge_index_nn, x=x_combined, edge_weight=edge_weight_nn, message_type='nn', max_degree_mask=mask_nn, size=None)
        out_ne = self.propagate(edge_index_ne, x=x_combined, edge_weight=edge_weight_ne, message_type='ne', max_degree_mask=mask_ne, size=None)
        out_ee = self.propagate(edge_index_ee, x=x_combined, edge_weight=edge_weight_ee, message_type='ee', max_degree_mask=mask_ee, size=None)
        out_en = self.propagate(edge_index_en, x=x_combined, edge_weight=edge_weight_en, message_type='en', max_degree_mask=mask_en, size=None)
        out_nexp = x_combined + out_nn + out_ne + out_en
        out_nexp = out_nexp[~max_degree_mask][:,:self.nexp_dim]
        out_exp =  x_combined + out_nn + out_ne + out_en + out_ee
        out_exp = out_exp[max_degree_mask]
      
        return out_exp, out_nexp

    def message(self, x_i: Tensor, x_j: Tensor, edge_index: OptTensor, edge_weight: OptTensor, message_type: str, max_degree_mask: Tensor) -> Tensor:
        if message_type == 'nn' or message_type == 'ee':
            pass
        elif message_type == 'ne':
            x_j_temp = x_j[:,:self.out_channels]
            x_j = self.lin_ne_expansion(x_j_temp)
        elif message_type == 'en':
            x_i_temp = x_i[:,:self.out_channels]
            combined = torch.concat([x_i_temp, x_j],dim=1)
            scores = torch.softmax(torch.relu(self.select(combined)),dim=-1)
            x_j = torch.gather(x_j, 1, torch.topk(scores, self.out_channels, dim=1)[1])
            paddings = torch.zeros(x_j.size(0), self.out_channels_max_degree-self.out_channels, device=x_j.device)
            x_j = torch.cat([x_j, paddings], dim=-1)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class PandaGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_channels_exp, out_channels_exp, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.in_channels_max_degree = in_channels_exp
        self.out_channels = out_channels
        self.out_channels_max_degree = out_channels_exp
        self.exp_dim = out_channels_exp
        self.nexp_dim = out_channels
        self.improved = False
        self.add_self_loops = True

        self.nn_nexp = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                      torch.nn.BatchNorm1d(out_channels), 
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(out_channels, out_channels))
        self.nn_exp = torch.nn.Sequential(torch.nn.Linear(in_channels_exp, out_channels_exp),
                                      torch.nn.BatchNorm1d(out_channels_exp), 
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(out_channels_exp, out_channels_exp))
        eps = 0.
        self.initial_eps = eps
        train_eps = False
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if out_channels >= in_channels:
            self.select = torch.nn.Linear(out_channels+out_channels_exp, out_channels_exp)
            self.lin_ne_expansion = torch.nn.Linear(out_channels, out_channels_exp)
        elif out_channels < in_channels:
            self.select = torch.nn.Linear(in_channels+in_channels_exp, in_channels_exp)
            self.lin_ne_expansion = torch.nn.Linear(in_channels, in_channels_exp)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.nn_nexp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.nn_exp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

        torch.nn.init.xavier_uniform_(self.select.weight)
        torch.nn.init.xavier_uniform_(self.lin_ne_expansion.weight)

    def forward(self, x_exp, x_nexp, edge_index, max_degree_mask, edge_weight=None):
        
        
        top_k_nodes = max_degree_mask.nonzero().to(x_nexp.device)
        mask_en = torch.isin(edge_index[0], top_k_nodes) & ~torch.isin(edge_index[1], top_k_nodes)
        mask_ne = torch.isin(edge_index[1], top_k_nodes) & ~torch.isin(edge_index[0], top_k_nodes)
        mask_nn = ~torch.any(torch.isin(edge_index, top_k_nodes), dim=0)
        mask_ee = torch.isin(edge_index[0], top_k_nodes) & torch.isin(edge_index[1], top_k_nodes)

        edge_index_nn = edge_index[:, mask_nn]
        edge_index_ee = edge_index[:, mask_ee]
        edge_index_en = edge_index[:, mask_en]
        edge_index_ne = edge_index[:, mask_ne]

        node_dim = x_nexp.shape[0]+x_exp.shape[0]
        
        x = torch.zeros((node_dim, x_exp.shape[-1]), dtype=x_nexp.dtype, device=x_nexp.device)
        if x_nexp.shape[-1] == x_exp.shape[-1]:
            x[~max_degree_mask] = x_nexp
            x[max_degree_mask] = x_exp
        else:
            padding_size = x_exp.shape[-1] - x_nexp.shape[-1]
            padding = torch.zeros(x_nexp.size(0), padding_size, device=x_nexp.device)
            x_nexp_padded = torch.cat([x_nexp, padding], dim=-1)
            x[~max_degree_mask] = x_nexp_padded
            x[max_degree_mask] = x_exp

        # # Place the results back according to the original node order
        x: OptPairTensor = (x, x)
        out_nn = self.propagate(edge_index_nn, x=x, message_type='nn')
        if x_nexp.shape[-1] == x_exp.shape[-1]:
            out_ne = self.propagate(edge_index_ne, x=x, message_type='nn')
            out_ee = self.propagate(edge_index_ee, x=x, message_type='ee')
            out_en = self.propagate(edge_index_en, x=x, message_type='ee')
            
            
            out_exp = out_nn + out_ne + out_ee + (1 + self.eps) * x[1]
            out_nexp = out_nn + out_ne + out_ee + out_en + (1 + self.eps) * x[1]
            out_exp = self.nn_exp(out_exp[max_degree_mask])
            out_nexp = self.nn_nexp(out_nexp[~max_degree_mask])
        else:
            out_ne = self.propagate(edge_index_ne, x=x, message_type='ne')
            out_ee = self.propagate(edge_index_ee, x=x, message_type='ee')
            out_en = self.propagate(edge_index_en, x=x, message_type='en')

            out_exp = out_nn + out_ne + out_ee + (1 + self.eps) * x[1]
            out_nexp = out_nn + out_ne + out_ee + out_en + (1 + self.eps) * x[1]
            out_exp = self.nn_exp(out_exp[max_degree_mask])
            if self.in_channels <= self.out_channels:
                out_nexp = self.nn_nexp(out_nexp[~max_degree_mask][:,:self.nexp_dim])
            elif self.in_channels > self.out_channels:
                out_nexp = self.nn_nexp(out_nexp[~max_degree_mask][:,:self.in_channels])

        return out_exp, out_nexp

    def message(self, x_i: Tensor, x_j: Tensor, message_type: str) -> Tensor:
        if message_type == 'nn' or message_type == 'ee':
            pass
        elif message_type == 'ne':
            if self.in_channels <= self.out_channels:
                x_j_temp = x_j[:,:self.out_channels]
            elif self.in_channels > self.out_channels:
                x_j_temp = x_j[:,:self.in_channels]
            x_j = self.lin_ne_expansion(x_j_temp)
        elif message_type == 'en':
            if self.in_channels <= self.out_channels:
                x_i_temp = x_i[:,:self.out_channels]
            elif self.in_channels > self.out_channels:
                x_i_temp = x_i[:,:self.in_channels]
            combined = torch.concat([x_i_temp, x_j],dim=1)
            scores = torch.softmax(torch.relu(self.select(combined)),dim=-1)
            if self.in_channels <= self.out_channels:
                x_j = torch.gather(x_j, 1, torch.topk(scores, self.out_channels, dim=1)[1])
                paddings = torch.zeros(x_j.size(0), self.out_channels_max_degree-self.out_channels, device=x_j.device)
            elif self.in_channels > self.out_channels:
                x_j = torch.gather(x_j, 1, torch.topk(scores, self.in_channels, dim=1)[1])
                paddings = torch.zeros(x_j.size(0), self.in_channels_max_degree-self.in_channels, device=x_j.device)
            x_j = torch.cat([x_j, paddings], dim=-1)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
    

class PandaRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_channels_exp, out_channels_exp, num_relations, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.in_channels_max_degree = in_channels_exp
        self.out_channels = out_channels
        self.out_channels_max_degree = out_channels_exp
        self.exp_dim = out_channels_exp
        self.nexp_dim = out_channels
        self.improved = False
        self.add_self_loops = True
        self.num_relations = num_relations

        self.weight = torch.nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.weight_max_degree = torch.nn.Parameter(torch.Tensor(num_relations, in_channels_exp, out_channels_exp))
        self.root = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.root_max_degree = torch.nn.Parameter(torch.Tensor(in_channels_exp, out_channels_exp))
        if out_channels >= in_channels:
            self.select = torch.nn.Linear(out_channels+out_channels_exp, out_channels_exp)
            self.lin_ne_expansion = torch.nn.Linear(out_channels, out_channels_exp)
        elif out_channels < in_channels:
            self.select = torch.nn.Linear(in_channels+in_channels_exp, in_channels_exp)
            self.lin_ne_expansion = torch.nn.Linear(in_channels, in_channels_exp)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight_max_degree)
        torch.nn.init.xavier_uniform_(self.root)
        torch.nn.init.xavier_uniform_(self.root_max_degree)
        torch.nn.init.xavier_uniform_(self.select.weight)
        torch.nn.init.xavier_uniform_(self.lin_ne_expansion.weight)

    def forward(self, x_exp, x_nexp, edge_index, max_degree_mask, edge_weight=None, edge_type=None):
        top_k_nodes = max_degree_mask.nonzero().to(x_nexp.device)
        mask_en = torch.isin(edge_index[0], top_k_nodes) & ~torch.isin(edge_index[1], top_k_nodes)
        mask_ne = torch.isin(edge_index[1], top_k_nodes) & ~torch.isin(edge_index[0], top_k_nodes)
        mask_nn = ~torch.any(torch.isin(edge_index, top_k_nodes), dim=0)
        mask_ee = torch.isin(edge_index[0], top_k_nodes) & torch.isin(edge_index[1], top_k_nodes)

        edge_index_nn = edge_index[:, mask_nn]
        edge_index_ee = edge_index[:, mask_ee]
        edge_index_en = edge_index[:, mask_en]
        edge_index_ne = edge_index[:, mask_ne]

        node_dim = x_nexp.shape[0]+x_exp.shape[0]
        
        x = torch.zeros((node_dim, x_exp.shape[-1]), dtype=x_nexp.dtype, device=x_nexp.device)
        if x_nexp.shape[-1] == x_exp.shape[-1]:
            x[~max_degree_mask] = x_nexp
            x[max_degree_mask] = x_exp
        else:
            padding_size = x_exp.shape[-1] - x_nexp.shape[-1]
            padding = torch.zeros(x_nexp.size(0), padding_size, device=x_nexp.device)
            x_nexp_padded = torch.cat([x_nexp, padding], dim=-1)
            x[~max_degree_mask] = x_nexp_padded
            x[max_degree_mask] = x_exp
        
        out_exp = torch.zeros((x_exp.shape[0], self.out_channels_max_degree), dtype=x_exp.dtype, device=x_exp.device)
        out_nexp = torch.zeros((x_nexp.shape[0], self.out_channels), dtype=x_nexp.dtype, device=x_nexp.device)

        for i in range(self.num_relations):
            out_nn = self.propagate(edge_index_nn, x=x, message_type='nn')
            if x_nexp.shape[-1] == x_exp.shape[-1]:
                out_ne = self.propagate(edge_index_ne, x=x, message_type='nn')
                out_ee = self.propagate(edge_index_ee, x=x, message_type='ee')
                out_en = self.propagate(edge_index_en, x=x, message_type='ee')
                
                h_nexp = out_nn + out_ne + out_en
                out_nexp = out_nexp + h_nexp[~max_degree_mask] @ self.weight[i]
                
                h_exp =  out_nn + out_ne + out_en + out_ee
                out_exp = out_exp + h_exp[max_degree_mask] @ self.weight_max_degree[i]
            else:
                out_ne = self.propagate(edge_index_ne, x=x, message_type='ne')
                out_ee = self.propagate(edge_index_ee, x=x, message_type='ee')
                out_en = self.propagate(edge_index_en, x=x, message_type='en')
                
                h_exp =  out_nn + out_ne + out_en + out_ee
                out_exp = out_exp + h_exp[max_degree_mask] @ self.weight_max_degree[i]
                
                h_nexp = out_nn + out_ne + out_en
                if self.in_channels <= self.out_channels:
                    out_nexp = out_nexp + h_nexp[~max_degree_mask][:,:self.nexp_dim] @ self.weight[i]
                elif self.in_channels > self.out_channels:
                    out_nexp = out_nexp + h_nexp[~max_degree_mask][:,:self.in_channels] @ self.weight[i]
        out_nexp += x_nexp @ self.root
        out_exp += x_exp @ self.root_max_degree

        return out_exp, out_nexp

    def message(self, x_i: Tensor, x_j: Tensor, message_type: str) -> Tensor:
        if message_type == 'nn' or message_type == 'ee':
            pass
        elif message_type == 'ne':
            if self.in_channels <= self.out_channels:
                x_j_temp = x_j[:,:self.out_channels]
            elif self.in_channels > self.out_channels:
                x_j_temp = x_j[:,:self.in_channels]
            x_j = self.lin_ne_expansion(x_j_temp)
        elif message_type == 'en':
            if self.in_channels <= self.out_channels:
                x_i_temp = x_i[:,:self.out_channels]
            elif self.in_channels > self.out_channels:
                x_i_temp = x_i[:,:self.in_channels]
            combined = torch.concat([x_i_temp, x_j],dim=1)
            scores = torch.softmax(torch.relu(self.select(combined)),dim=-1)
            if self.in_channels <= self.out_channels:
                x_j = torch.gather(x_j, 1, torch.topk(scores, self.out_channels, dim=1)[1])
                paddings = torch.zeros(x_j.size(0), self.out_channels_max_degree-self.out_channels, device=x_j.device)
            elif self.in_channels > self.out_channels:
                x_j = torch.gather(x_j, 1, torch.topk(scores, self.in_channels, dim=1)[1])
                paddings = torch.zeros(x_j.size(0), self.in_channels_max_degree-self.in_channels, device=x_j.device)
            x_j = torch.cat([x_j, paddings], dim=-1)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class PandaRGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_channels_exp, out_channels_exp, num_relations, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.in_channels_max_degree = in_channels_exp
        self.out_channels = out_channels
        self.out_channels_max_degree = out_channels_exp
        self.exp_dim = out_channels_exp
        self.nexp_dim = out_channels
        self.num_relations = num_relations
        self.improved = False
        self.add_self_loops = True

        nns_nexp = []
        nns_exp = []
        for i in range(self.num_relations):
            nns_nexp.append(torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                      torch.nn.BatchNorm1d(out_channels), 
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(out_channels, out_channels))
                            )
            nns_exp.append(torch.nn.Sequential(torch.nn.Linear(in_channels_exp, out_channels_exp),
                                      torch.nn.BatchNorm1d(out_channels_exp), 
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(out_channels_exp, out_channels_exp))
                            )
        self.nn_nexp = ModuleList(nns_nexp)
        self.nn_exp = ModuleList(nns_exp)
        self.self_loop_conv_nexp = torch.nn.Linear(in_channels, out_channels)
        self.self_loop_conv_exp = torch.nn.Linear(in_channels_exp, out_channels_exp)

        eps = 0.
        self.initial_eps = eps
        train_eps = False
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if out_channels >= in_channels:
            self.select = torch.nn.Linear(out_channels+out_channels_exp, out_channels_exp)
            self.lin_ne_expansion = torch.nn.Linear(out_channels, out_channels_exp)
        elif out_channels < in_channels:
            self.select = torch.nn.Linear(in_channels+in_channels_exp, in_channels_exp)
            self.lin_ne_expansion = torch.nn.Linear(in_channels, in_channels_exp)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.nn_nexp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.nn_exp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

        torch.nn.init.xavier_uniform_(self.select.weight)
        torch.nn.init.xavier_uniform_(self.lin_ne_expansion.weight)

    def forward(self, x_exp, x_nexp, edge_index, max_degree_mask, edge_weight=None, edge_type=None):
        
        
        top_k_nodes = max_degree_mask.nonzero().to(x_nexp.device)
        mask_en = torch.isin(edge_index[0], top_k_nodes) & ~torch.isin(edge_index[1], top_k_nodes)
        mask_ne = torch.isin(edge_index[1], top_k_nodes) & ~torch.isin(edge_index[0], top_k_nodes)
        mask_nn = ~torch.any(torch.isin(edge_index, top_k_nodes), dim=0)
        mask_ee = torch.isin(edge_index[0], top_k_nodes) & torch.isin(edge_index[1], top_k_nodes)

        edge_index_nn = edge_index[:, mask_nn]
        edge_index_ee = edge_index[:, mask_ee]
        edge_index_en = edge_index[:, mask_en]
        edge_index_ne = edge_index[:, mask_ne]

        node_dim = x_nexp.shape[0]+x_exp.shape[0]
        
        x = torch.zeros((node_dim, x_exp.shape[-1]), dtype=x_nexp.dtype, device=x_nexp.device)
        if x_nexp.shape[-1] == x_exp.shape[-1]:
            x[~max_degree_mask] = x_nexp
            x[max_degree_mask] = x_exp
        else:
            padding_size = x_exp.shape[-1] - x_nexp.shape[-1]
            padding = torch.zeros(x_nexp.size(0), padding_size, device=x_nexp.device)
            x_nexp_padded = torch.cat([x_nexp, padding], dim=-1)
            x[~max_degree_mask] = x_nexp_padded
            x[max_degree_mask] = x_exp

        # # Place the results back according to the original node order
        x: OptPairTensor = (x, x)
        x_new_nexp = self.self_loop_conv_nexp(x_nexp)
        x_new_exp = self.self_loop_conv_exp(x_exp)
        for i in range(self.num_relations):
            out_nn = self.propagate(edge_index_nn, x=x, message_type='nn')
            if x_nexp.shape[-1] == x_exp.shape[-1]:
                out_ne = self.propagate(edge_index_ne, x=x, message_type='nn')
                out_ee = self.propagate(edge_index_ee, x=x, message_type='ee')
                out_en = self.propagate(edge_index_en, x=x, message_type='ee')
                   
                out_exp = out_nn + out_ne + out_ee + (1 + self.eps) * x[1]
                out_nexp = out_nn + out_ne + out_ee + out_en + (1 + self.eps) * x[1]
                out_exp = self.nn_exp[i](out_exp[max_degree_mask])
                out_nexp = self.nn_nexp[i](out_nexp[~max_degree_mask])
            else:
                out_ne = self.propagate(edge_index_ne, x=x, message_type='ne')
                out_ee = self.propagate(edge_index_ee, x=x, message_type='ee')
                out_en = self.propagate(edge_index_en, x=x, message_type='en')

                out_exp = out_nn + out_ne + out_ee + (1 + self.eps) * x[1]
                out_nexp = out_nn + out_ne + out_ee + out_en + (1 + self.eps) * x[1]
                out_exp = self.nn_exp[i](out_exp[max_degree_mask])
                if self.in_channels <= self.out_channels:
                    out_nexp = self.nn_nexp[i](out_nexp[~max_degree_mask][:,:self.nexp_dim])
                elif self.in_channels > self.out_channels:
                    out_nexp = self.nn_nexp[i](out_nexp[~max_degree_mask][:,:self.in_channels])
            out_nexp = x_new_nexp + out_nexp
            out_exp = x_new_exp + out_exp
        return out_exp, out_nexp

    def message(self, x_i: Tensor, x_j: Tensor, message_type: str) -> Tensor:
        if message_type == 'nn' or message_type == 'ee':
            pass
        elif message_type == 'ne':
            if self.in_channels <= self.out_channels:
                x_j_temp = x_j[:,:self.out_channels]
            elif self.in_channels > self.out_channels:
                x_j_temp = x_j[:,:self.in_channels]
            x_j = self.lin_ne_expansion(x_j_temp)
        elif message_type == 'en':
            if self.in_channels <= self.out_channels:
                x_i_temp = x_i[:,:self.out_channels]
            elif self.in_channels > self.out_channels:
                x_i_temp = x_i[:,:self.in_channels]
            combined = torch.concat([x_i_temp, x_j],dim=1)
            scores = torch.softmax(torch.relu(self.select(combined)),dim=-1)
            if self.in_channels <= self.out_channels:
                x_j = torch.gather(x_j, 1, torch.topk(scores, self.out_channels, dim=1)[1])
                paddings = torch.zeros(x_j.size(0), self.out_channels_max_degree-self.out_channels, device=x_j.device)
            elif self.in_channels > self.out_channels:
                x_j = torch.gather(x_j, 1, torch.topk(scores, self.in_channels, dim=1)[1])
                paddings = torch.zeros(x_j.size(0), self.in_channels_max_degree-self.in_channels, device=x_j.device)
            x_j = torch.cat([x_j, paddings], dim=-1)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
    