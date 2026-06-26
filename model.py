import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import softmax as pyg_softmax, add_self_loops


class GCNBranch(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATBranch(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=8,
                 dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                        dropout=dropout))
        self.convs.append(
            GATConv(hidden_dim, out_dim, heads=1, concat=False,
                    dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCNGATHybrid(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Q_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, h_gcn, h_gat, edge_index):
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=h_gcn.size(0))
        src, dst = edge_index_sl
        D = self.embed_dim
        N = h_gcn.size(0)

        Q_g = self.W_Q(h_gcn)
        K_a = self.W_K(h_gat)
        V_a = self.W_V(h_gat)

        Q_a = self.W_Q_prime(h_gat)
        K_g = self.W_K_prime(h_gcn)
        V_g = self.W_V_prime(h_gcn)

        score_g2a = (Q_g[dst] * K_a[src]).sum(dim=-1) / (D ** 0.5)
        alpha_g2a = pyg_softmax(score_g2a, dst, num_nodes=N)
        agg_g2a = torch.zeros(N, D, device=h_gcn.device)
        agg_g2a.scatter_add_(
            0, dst.unsqueeze(1).expand(-1, D), alpha_g2a.unsqueeze(1) * V_a[src])

        score_a2g = (Q_a[dst] * K_g[src]).sum(dim=-1) / (D ** 0.5)
        alpha_a2g = pyg_softmax(score_a2g, dst, num_nodes=N)
        agg_a2g = torch.zeros(N, D, device=h_gcn.device)
        agg_a2g.scatter_add_(
            0, dst.unsqueeze(1).expand(-1, D), alpha_a2g.unsqueeze(1) * V_g[src])

        h_cross = F.relu(agg_g2a + agg_a2g)
        h_hybrid = self.layer_norm(h_cross + h_gcn + h_gat)
        return h_hybrid


class FCMAC(nn.Module):

    def __init__(self, embed_dim, K_gcn=34, K_gat=33, K_gcn_gat=33, D_z=1):
        super().__init__()
        self.K_gcn = K_gcn
        self.K_gat = K_gat
        self.K_gcn_gat = K_gcn_gat
        self.D_z = D_z

        self.c_gcn = nn.Parameter(torch.randn(K_gcn, embed_dim) * 0.1)
        self.c_gat = nn.Parameter(torch.randn(K_gat, embed_dim) * 0.1)
        self.c_gcn_gat = nn.Parameter(torch.randn(K_gcn_gat, embed_dim) * 0.1)

        self.delta_gcn = nn.Parameter(torch.ones(K_gcn))
        self.delta_gat = nn.Parameter(torch.ones(K_gat))
        self.delta_gcn_gat = nn.Parameter(torch.ones(K_gcn_gat))

        self.attn_fc = nn.Linear(2, 1)

        self.w = nn.Parameter(
            torch.randn(K_gcn, K_gat, K_gcn_gat, 2, D_z) * 0.01)

    def _compute_membership(self, h, centers, deltas):
        diff = h.unsqueeze(1) - centers.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-dist_sq / (2.0 * deltas.unsqueeze(0) ** 2 + 1e-8))

    def forward(self, h_gcn, h_gat, h_gcn_gat):
        mu_gcn = self._compute_membership(h_gcn, self.c_gcn, self.delta_gcn)
        mu_gat = self._compute_membership(h_gat, self.c_gat, self.delta_gat)
        mu_gg = self._compute_membership(
            h_gcn_gat, self.c_gcn_gat, self.delta_gcn_gat)

        r_m0 = (mu_gcn.unsqueeze(2).unsqueeze(3)
                * mu_gat.unsqueeze(1).unsqueeze(3)
                * mu_gg.unsqueeze(1).unsqueeze(2))
        r_m1 = (mu_gcn.unsqueeze(2).unsqueeze(3)
                * (1 - mu_gat).unsqueeze(1).unsqueeze(3)
                * (1 - mu_gg).unsqueeze(1).unsqueeze(2))

        avg_m0 = r_m0.mean(dim=0)
        avg_m1 = r_m1.mean(dim=0)
        u_input = torch.stack([avg_m0, avg_m1], dim=-1)
        u = self.attn_fc(u_input).squeeze(-1)
        beta = F.softmax(u.reshape(-1), dim=0).reshape(
            self.K_gcn, self.K_gat, self.K_gcn_gat)

        z = torch.einsum('nklq,klq,klqd->nd', r_m0, beta, self.w[:, :, :, 0, :])
        z = z + torch.einsum('nklq,klq,klqd->nd', r_m1, beta, self.w[:, :, :, 1, :])

        return z

    @torch.no_grad()
    def initialize_centers(self, h_gcn_all, h_gat_all, h_gcn_gat_all):
        from sklearn.cluster import KMeans

        for h_all, centers, deltas, K in [
            (h_gcn_all, self.c_gcn, self.delta_gcn, self.K_gcn),
            (h_gat_all, self.c_gat, self.delta_gat, self.K_gat),
            (h_gcn_gat_all, self.c_gcn_gat, self.delta_gcn_gat, self.K_gcn_gat),
        ]:
            h_np = h_all.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(h_np)
            centers.data.copy_(
                torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
            for k in range(K):
                mask = kmeans.labels_ == k
                if mask.sum() > 0:
                    dists = np.sqrt(
                        np.sum((h_np[mask] - kmeans.cluster_centers_[k]) ** 2,
                               axis=1))
                    deltas.data[k] = max(float(dists.mean()), 1e-3)


class GCATFCMAC(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.gcn = GCNBranch(
            cfg.num_node_features, D, D,
            num_layers=cfg.gcn_layers, dropout=cfg.dropout)
        self.gat = GATBranch(
            cfg.num_node_features, D, D,
            num_layers=cfg.gat_layers, heads=cfg.gat_heads,
            dropout=cfg.dropout)
        self.hybrid = GCNGATHybrid(D)

        self.fcmac = FCMAC(
            embed_dim=D,
            K_gcn=cfg.K_gcn, K_gat=cfg.K_gat, K_gcn_gat=cfg.K_gcn_gat,
            D_z=cfg.fcmac_output_dim)

        self.localizer = nn.Sequential(
            nn.Linear(cfg.fcmac_output_dim, cfg.localizer_hidden),
            nn.ReLU(),
            nn.Linear(cfg.localizer_hidden, 2),
        )

        self.recon_gcn = nn.Sequential(nn.Linear(D, D), nn.ReLU())
        self.recon_gat = nn.Sequential(nn.Linear(D, D), nn.ReLU())
        self.recon_gcn_gat = nn.Sequential(nn.Linear(D, D), nn.ReLU())

    def forward(self, x, edge_index):
        h_gcn = self.gcn(x, edge_index)
        h_gat = self.gat(x, edge_index)
        h_gcn_gat = self.hybrid(h_gcn, h_gat, edge_index)
        z = self.fcmac(h_gcn, h_gat, h_gcn_gat)
        p_hat = self.localizer(z)

        h_gcn_recon = self.recon_gcn(h_gcn)
        h_gat_recon = self.recon_gat(h_gat)
        h_gcn_gat_recon = self.recon_gcn_gat(h_gcn_gat)

        return p_hat, h_gcn, h_gat, h_gcn_gat, h_gcn_recon, h_gat_recon, h_gcn_gat_recon

    def get_embeddings(self, x, edge_index):
        h_gcn = self.gcn(x, edge_index)
        h_gat = self.gat(x, edge_index)
        h_gcn_gat = self.hybrid(h_gcn, h_gat, edge_index)
        return h_gcn, h_gat, h_gcn_gat


def compute_loss(p_hat, p_true, intrusion_mask,
                 h_gcn, h_gat, h_gcn_gat,
                 h_gcn_recon, h_gat_recon, h_gcn_gat_recon,
                 model, lam=0.001, alpha=0.1):
    if intrusion_mask.sum() > 0:
        loc_loss = F.mse_loss(p_hat[intrusion_mask], p_true[intrusion_mask])
    else:
        loc_loss = torch.tensor(0.0, device=p_hat.device)

    reg_loss = sum(p.pow(2).sum() for p in model.parameters())
    L_theta = loc_loss + lam * reg_loss

    L_gcn = F.mse_loss(h_gcn_recon, h_gcn.detach())
    L_gat = F.mse_loss(h_gat_recon, h_gat.detach())
    L_gcn_gat = F.mse_loss(h_gcn_gat_recon, h_gcn_gat.detach())

    total_loss = L_theta + alpha * (L_gcn + L_gat + L_gcn_gat)
    return total_loss, loc_loss.item()
