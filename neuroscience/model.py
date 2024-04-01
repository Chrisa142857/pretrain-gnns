import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv
# from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
# from loader import BioDataset
# from dataloader import DataLoaderFinetune
# from torch_scatter import scatter_add
# from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj

class MLP_batchnorm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden=1, output_activation='relu',batchnorm_affine=True, device='cpu', dtype=torch.float32):
        super().__init__()
        # Inputs to hidden layer linear transformation
        assert num_hidden > 0
        self.num_hidden = num_hidden
        self.dtype = dtype
        
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.linears.append(nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype))
        self.bns.append(nn.BatchNorm1d(hidden_dim, affine=batchnorm_affine, device=device, dtype=dtype))
        
        for layer in range(num_hidden-1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype))
            self.bns.append(nn.BatchNorm1d(hidden_dim, affine=batchnorm_affine, device=device, dtype=dtype))
        self.linears.append(nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype))
        self.activation = nn.functional.relu
        if output_activation == 'relu':
            self.bns.append(nn.BatchNorm1d(output_dim, affine=batchnorm_affine, device=device, dtype=dtype)) 
            self.output_activation = nn.functional.relu
        elif output_activation == 'linear':
            self.output_activation = None
        else:
            raise 'unknown activation for output layer of MLP'
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        
        for layer in range(self.num_hidden):
            x = self.linears[layer](x)
            x = self.bns[layer](x)
            x = self.activation(x)
            
        x = self.linears[-1](x)
        if not (self.output_activation) is None:
            x = self.bns[-1](x)
            x = self.output_activation(x)
        return x
    


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "GINConv", init_connect = True):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        gnn_conv = getattr(torch_geometric.nn, gnn_type)
        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False
            
            if gnn_type == "GINConv":
                self.gnns.append(gnn_conv(MLP_batchnorm(emb_dim, emb_dim, emb_dim)))
            else:
                self.gnns.append(gnn_conv(emb_dim, emb_dim))
        self.init_connect = init_connect
    #def forward(self, x, edge_index, edge_attr):
    # def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_adj: Adj, prompt=None):
        h_list = [x]
        for layer in range(self.num_layer):
            if prompt is not None:
                h_list[layer] = prompt.add(h_list[layer])
            h = self.gnns[layer](h_list[layer], edge_adj)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if self.init_connect:
                h = h + x
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, in_dim, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.encoder = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim))
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data, prompt=None):
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x, edge_adj, batch = data.x, data.edge_index, data.batch
        node_representation = self.gnn(x, edge_adj, prompt)

        pooled = self.pool(node_representation, batch)
        graph_rep = pooled
        # center_node_rep = node_representation[data.center_node_idx]

        # graph_rep = torch.cat([pooled, center_node_rep], dim = 1)

        return self.graph_pred_linear(graph_rep)


if __name__ == "__main__":
    pass



