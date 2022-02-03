import torch
import torch.nn as nn
import torch.nn.functional as F
from config import _C as C
from models.layers.GNN_dmwater import GraphNet
from scipy import spatial
import numpy as np
import utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_dim_in = C.NET.NODE_FEAT_DIM_IN
        self.edge_dim_in = C.NET.EDGE_FEAT_DIM_IN

        self.hidden_size = C.NET.HIDDEN_SIZE
        self.out_size = C.NET.OUT_SIZE
        num_layers = C.NET.GNN_LAYER
        
        self.particle_emb = nn.Embedding(C.NUM_PARTICLE_TYPES, C.NET.PARTICLE_EMB_SIZE)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.graph = GraphNet(layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size),
        )

    def _construct_graph_nodes(self, poss, particle_type, metadata):
        vels = utils.time_diff(poss)
        vels = (vels - metadata['vel_mean'])/metadata['vel_std']
        n_vel, d_vel = vels.shape[1], vels.shape[2]
        assert n_vel == C.N_HIS - 1
        vels = vels.reshape([-1, n_vel*d_vel])

        pos_last = poss[:, -1]
        dist_to_walls = torch.cat(
                        [pos_last - metadata['bounds'][:, 0],
                        -pos_last + metadata['bounds'][:, 1]], 1)
        dist_to_walls = torch.clip(dist_to_walls/C.NET.RADIUS, -1, 1) 

        type_emb = self.particle_emb(particle_type)

        node_attr = torch.cat([vels,
                               dist_to_walls,
                               type_emb], axis=1)

        return node_attr

    def _construct_graph_edges(self, pos):
        device = pos.device
        collapsed = False

        n_particles = pos.shape[0]
        # Calculate undirected edge list using KDTree
        point_tree = spatial.cKDTree(pos.detach().cpu().numpy())
        undirected_pairs = np.array(list(point_tree.query_pairs(C.NET.RADIUS, p=2))).T
        undirected_pairs = torch.from_numpy(undirected_pairs).to(device)
        pairs = torch.cat([undirected_pairs, torch.flip(undirected_pairs, dims=(0,))], dim=1).long()

        if C.NET.SELF_EDGE:
            self_pairs = torch.stack([torch.arange(n_particles, device=device), 
                                    torch.arange(n_particles, device=device)])
            pairs = torch.cat([pairs, self_pairs], dim=1)

        # check if prediction collapsed in long term unrolling
        if pairs.shape[1] > C.NET.MAX_EDGE_PER_PARTICLE * n_particles:
            collapsed = True

        senders = pairs[0]
        receivers = pairs[1]

        # Calculate corresponding relative edge attributes (distance vector + magnitude)
        dist_vec = (pos[senders] - pos[receivers])
        dist_vec = dist_vec / C.NET.RADIUS
        dist = torch.linalg.norm(dist_vec, dim=1, keepdims=True)
        edges = torch.cat([dist_vec, dist], dim=1)

        return edges, senders, receivers, collapsed

    def forward(self, poss, particle_type, metadata, nonk_mask, tgt_poss, num_rollouts=10, phase='train'):

        pred_accns = []
        pred_poss = []
        for i in range(num_rollouts):
            nodes = self._construct_graph_nodes(poss, particle_type, metadata)
            edges, senders, receivers, collapsed = self._construct_graph_edges(poss[:, -1])

            nodes = self.node_encoder(nodes)
            edges = self.edge_encoder(edges)

            nodes, edges = self.graph(nodes, edges, senders, receivers)

            pred_accn = self.decoder(nodes)
            pred_acc = pred_accn * metadata['acc_std'] + metadata['acc_mean']
            pred_accns.append(pred_accn)

            prev_vel = poss[:, -1] - poss[:, -2]
            pred_pos = poss[:, -1] + prev_vel + pred_acc
                
            # replace kinematic nodes
            pred_pos = torch.where(nonk_mask[:, None].bool(), pred_pos, tgt_poss[:, i])
            poss = torch.cat([poss[:, 1:], pred_pos[:, None]], dim=1)
            pred_poss.append(pred_pos)

            if collapsed:
                break

        pred_accns = torch.stack(pred_accns).permute(1, 0, 2)
        pred_poss = torch.stack(pred_poss).permute(1, 0, 2)

        outputs = {
            'pred_accns': pred_accns,
            'pred_poss': pred_poss,
            'pred_collaposed': collapsed
        }
        
        return outputs
  