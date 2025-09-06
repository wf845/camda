import torch
import torch as th
from torch import nn
import torch.nn.functional as F

# =====================================================================================
# CAMDA Components
# =====================================================================================

class GCNLayer(nn.Module):
    """ A single GCN layer with normalization. """
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, adj, features):
        adj_self_loop = adj + torch.eye(adj.shape[0], device=adj.device)
        D = torch.sum(adj_self_loop, 1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
        norm_adj = torch.matmul(torch.matmul(D_mat_inv_sqrt, adj_self_loop), D_mat_inv_sqrt)
        
        h = torch.matmul(norm_adj, features)
        return self.linear(h)

class GCN(nn.Module):
    """ A multi-layer GCN encoder. """
    def __init__(self, in_feats, out_feats, num_layers=2, dropout=0.1):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(GCNLayer(in_feats, out_feats))
        
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(out_feats, out_feats))
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        h = features
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(adj, h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
            
            if h.shape == h_in.shape:
                h = h + h_in
        return h

class Attention(nn.Module):
    """ Attention mechanism to fuse metapath-specific embeddings. """
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, mode='none'):
        if mode == 'no_attention':
            return z.mean(1)
        
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class Discriminator(nn.Module):
    """ Discriminator for contrastive learning. """
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, summary):
        """
        Bilinear discriminator: Dn(a,b) = σ(a^T W_D b)
        Output probability values in [0,1] range
        - features: (N, D) tensor of node features
        - summary: (N, D) tensor of summary features (can be node-specific)
        """
        weighted_features = torch.matmul(features, self.weight)
        score = torch.mean(weighted_features * summary, dim=1)
        return torch.sigmoid(score)


class ConfigurableMLP(nn.Module):
    """Configurable multi-layer MLP predictor"""
    def __init__(self, input_dim, layer_dims, dropout=0.2, activation='relu', batch_norm=False):
        super(ConfigurableMLP, self).__init__()
        
        if not layer_dims:
            raise ValueError("layer_dims cannot be empty")
        
        layers = []
        current_dim = input_dim
        
        for i, next_dim in enumerate(layer_dims):
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < len(layer_dims) - 1:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(next_dim))
                
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                elif activation.lower() == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation.lower() == 'leakyrelu':
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class TopologyView(nn.Module):
    """Topology view encoder (third view)"""
    def __init__(self, in_feats, out_feats, num_layers=2, dropout=0.1, restart_prob=0.2):
        super(TopologyView, self).__init__()
        self.restart_prob = restart_prob
        self.gcn = GCN(in_feats, out_feats, num_layers, dropout)
    
    def compute_topology_matrix(self, adj):
        """Compute simplified topology influence matrix"""
        D = torch.sum(adj, dim=1, keepdim=True) + 1e-8
        P = adj / D
        topology_matrix = self.restart_prob * adj + (1 - self.restart_prob) * torch.mm(P, adj)
        return topology_matrix
    
    def forward(self, adj, features):
        topology_adj = self.compute_topology_matrix(adj)
        return self.gcn(topology_adj, features)


# =====================================================================================
# CAMDA Main Model
# =====================================================================================

class CAMDA(nn.Module):
    def __init__(self, args, num_metapaths):
        super(CAMDA, self).__init__()
        self.args = args
        self.num_metapaths = num_metapaths
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lin_m = nn.Linear(args.miRNA_number, args.hid_feats)
        self.lin_d = nn.Linear(args.disease_number, args.hid_feats)

        dropout_rate = getattr(args, 'dropout', 0.1)

        self.personalized_encoders = nn.ModuleList([
            GCN(args.hid_feats, args.hid_feats, num_layers=args.gcn_layers, dropout=dropout_rate) 
            for _ in range(num_metapaths)
        ])

        self.shared_encoder = GCN(args.hid_feats, args.hid_feats, num_layers=args.gcn_layers, dropout=dropout_rate)

        self.topology_encoders = nn.ModuleList([
            TopologyView(args.hid_feats, args.hid_feats, num_layers=args.gcn_layers, dropout=dropout_rate)
            for _ in range(num_metapaths)
        ])

        self.personalized_attention = Attention(args.hid_feats)
        self.shared_attention = Attention(args.hid_feats)
        self.topology_attention = Attention(args.hid_feats)

        
        self.to(device)

    def forward(self, metapath_adjs, features, target_type):
        """
        Forward pass for one type of node (e.g., miRNA).
        Returns three view representations: Z(personalized), H(shared), S(topology)
        """
        if target_type == 'miRNA':
            projected_features = F.elu(self.lin_m(features))
        else:
            projected_features = F.elu(self.lin_d(features))

        personalized_embs = []
        for i, adj in enumerate(metapath_adjs):
            personalized_embs.append(self.personalized_encoders[i](adj, projected_features))
        personalized_embs = torch.stack(personalized_embs, dim=1)

        shared_embs = []
        for adj in metapath_adjs:
            shared_embs.append(self.shared_encoder(adj, projected_features))
        shared_embs = torch.stack(shared_embs, dim=1)

        topology_embs = []
        for i, adj in enumerate(metapath_adjs):
            topology_embs.append(self.topology_encoders[i](adj, projected_features))
        topology_embs = torch.stack(topology_embs, dim=1)

        Z = self.personalized_attention(personalized_embs)
        H = self.shared_attention(shared_embs)
        S = self.topology_attention(topology_embs)

        s_j = F.sigmoid(Z.mean(dim=0))
        g_j = F.sigmoid(H.mean(dim=0))
        t_j = F.sigmoid(S.mean(dim=0))

        return Z, H, S, s_j, g_j, t_j, personalized_embs, shared_embs, topology_embs