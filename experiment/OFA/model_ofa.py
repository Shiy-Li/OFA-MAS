import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

class MoELayer(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, hidden_dim=256):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, weights):
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expanded_weights = weights.unsqueeze(2)
        final_output = torch.sum(outputs * expanded_weights, dim=1)
        return final_output


class MAGNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, task_dim):
        super(MAGNetLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_m = nn.Linear(in_dim, out_dim)
        self.W_g = nn.Linear(in_dim + task_dim, out_dim)

        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_q = nn.Linear(out_dim, out_dim)
        self.a = nn.Parameter(torch.randn(out_dim * 2, 1))

    def forward(self, h, adj, z):
        num_nodes = h.size(0)

        raw_message = F.relu(self.W_m(h))
        gate = torch.sigmoid(self.W_g(torch.cat([h, z], dim=1)))
        gated_message = gate * raw_message

        gate_l1_loss = torch.norm(gate, p=1) / gate.numel()

        q = self.W_q(gated_message)
        k = self.W_k(h)

        q_expanded = q.unsqueeze(1).repeat(1, num_nodes, 1)
        k_expanded = k.unsqueeze(0).repeat(num_nodes, 1, 1)

        attn_input = torch.cat([k_expanded, q_expanded], dim=2)
        e = F.leaky_relu((attn_input @ self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        agg_message = attention.t() @ gated_message

        h_new = (agg_message + h) / 2

        return h_new, gate_l1_loss


class MAGNet(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, task_dim):
        super(MAGNet, self).__init__()
        self.initial_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MAGNetLayer(hidden_dim, hidden_dim, task_dim))

    def forward(self, h, adj, z):
        if h.shape[0] == 0:
            return h, torch.tensor(0.0, device=z.device)

        h = self.initial_proj(h)

        num_nodes = h.shape[0]
        if num_nodes > 0 and z.shape[0] != num_nodes:
            if z.shape[0] == 1:
                z = z.repeat(num_nodes, 1)
            else:
                raise ValueError("Shape of z is not compatible with h")

        total_gate_l1_loss = 0
        for layer in self.layers:
            h, gate_l1_loss = layer(h, adj, z)
            total_gate_l1_loss += gate_l1_loss
            
        return h, total_gate_l1_loss / len(self.layers)


class OFAModel(nn.Module):
    def __init__(self, args, role_embeddings_dict):
        super(OFAModel, self).__init__()
        self.args = args
        if not hasattr(self.args, 'min_nodes'):
            self.args.min_nodes = 2

        self.task_encoder = nn.Sequential(
            nn.Linear(args.embedding_size_question, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.task_embedding_dim)
        )

        self.role_names = list(role_embeddings_dict.keys())
        base_role_embeddings = torch.stack(list(role_embeddings_dict.values()), dim=0)

        start_embedding = torch.zeros(1, self.args.embedding_size_role)
        end_embedding = torch.zeros(1, self.args.embedding_size_role)
        pad_embedding = torch.zeros(1, self.args.embedding_size_role)
        full_embedding_matrix = torch.cat([base_role_embeddings, start_embedding, end_embedding, pad_embedding], dim=0)
        self.register_buffer('role_embeddings', full_embedding_matrix)

        self.id_to_role = {i: name for i, name in enumerate(self.role_names)}
        self.role_to_id = {name: i for i, name in enumerate(self.role_names)}

        self.START_TOKEN_ID = len(self.role_names)
        self.END_TOKEN_ID = len(self.role_names) + 1
        self.PAD_TOKEN_ID = len(self.role_names) + 2

        self.id_to_role[self.START_TOKEN_ID] = 'START_TOKEN'
        self.id_to_role[self.END_TOKEN_ID] = 'END_TOKEN'
        self.id_to_role[self.PAD_TOKEN_ID] = 'PAD_TOKEN'
        self.role_to_id['START_TOKEN'] = self.START_TOKEN_ID
        self.role_to_id['END_TOKEN'] = self.END_TOKEN_ID
        self.role_to_id['PAD_TOKEN'] = self.PAD_TOKEN_ID

        self.moe_gate = nn.Sequential(
            nn.Linear(args.task_embedding_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.num_experts),
            nn.Softmax(dim=-1)
        )

        self.magnet = MAGNet(
            num_layers=args.gcn_layers,
            in_dim=args.embedding_size_role,
            hidden_dim=args.hidden_dim,
            task_dim=args.task_embedding_dim
        )

        self.graph_readout = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU()
        )

        self.moe_query_net = MoELayer(
            num_experts=args.num_experts,
            input_dim=args.hidden_dim,
            output_dim=args.embedding_size_role
        )
        self.moe_edge_net = MoELayer(
            num_experts=args.num_experts,
            input_dim=args.hidden_dim + args.embedding_size_role + args.task_embedding_dim,
            output_dim=1
        )

    def forward(self, batch, beta_vae=1.0, unconditional=False):
        batch_size = batch['task_embedding'].shape[0]
        device = batch['task_embedding'].device

        task_embedding = batch['task_embedding']
        if unconditional:
            task_embedding = torch.zeros_like(task_embedding)

        z = self.task_encoder(task_embedding)

        moe_weights = self.moe_gate(z)
        f_k = torch.mean(moe_weights, dim=0)
        loss_balance = self.args.num_experts * torch.sum(f_k * f_k)

        adj_gt = batch['adj_gt']
        node_roles_gt = batch['node_roles']
        graph_sizes = batch['graph_sizes']
        max_nodes = adj_gt.shape[1]

        total_loss_graph = 0
        total_loss_gate_l1 = 0
        total_correct_predictions = 0
        total_node_predictions = 0

        t_range = torch.arange(1, max_nodes + 1, device=device).unsqueeze(0)
        active_mask = (t_range <= graph_sizes.unsqueeze(1)).float()

        for t in range(1, max_nodes):
            num_active_graphs = int(active_mask[:, t].sum())
            if num_active_graphs == 0:
                continue

            active_indices = torch.where(active_mask[:, t] == 1)[0]
            
            sub_adjs = adj_gt[active_indices, :t - 1, :t - 1]
            batched_adj = torch.block_diag(*sub_adjs)
            
            sub_role_ids = node_roles_gt[active_indices, :t - 1].reshape(-1)
            valid_node_indices = torch.where(sub_role_ids != self.PAD_TOKEN_ID)[0]
            
            h_existing = self.role_embeddings[sub_role_ids[valid_node_indices]]
            
            batched_adj = batched_adj[valid_node_indices][:, valid_node_indices]

            z_active = z[active_indices]
            node_to_graph_map = torch.arange(num_active_graphs, device=device).repeat_interleave(t - 1)
            z_repeated = z_active[node_to_graph_map[valid_node_indices]]

            h_encoded, gate_l1_loss_step = self.magnet(h_existing, batched_adj, z_repeated)
            
            g_t_minus_1_list = []
            current_pos = 0
            for i in range(num_active_graphs):
                num_nodes_in_graph = (node_roles_gt[active_indices[i], :t - 1] != self.PAD_TOKEN_ID).sum()
                if num_nodes_in_graph > 0:
                    graph_h = h_encoded[current_pos : current_pos + num_nodes_in_graph]
                    g_t_minus_1_list.append(self.graph_readout(torch.mean(graph_h, dim=0)))
                    current_pos += num_nodes_in_graph
                else:
                    g_t_minus_1_list.append(torch.zeros(self.args.hidden_dim, device=device))
            
            g_t_minus_1 = torch.stack(g_t_minus_1_list)

            q_t = self.moe_query_net(g_t_minus_1, moe_weights[active_indices])
            node_logits = q_t @ self.role_embeddings.t()
            gt_node_ids = node_roles_gt[active_indices, t - 1]

            loss_node = F.cross_entropy(node_logits, gt_node_ids, reduction='none')
            total_loss_graph += loss_node.sum()

            pred_node_ids = torch.argmax(node_logits, dim=1)
            total_correct_predictions += (pred_node_ids == gt_node_ids).sum().item()
            total_node_predictions += num_active_graphs

            gt_new_node_embeddings = self.role_embeddings[gt_node_ids]
            valid_edge_pred_mask = (gt_node_ids != self.END_TOKEN_ID) & (gt_node_ids != self.PAD_TOKEN_ID)
            
            if t > 1 and valid_edge_pred_mask.any():
                edge_pred_indices = torch.where(valid_edge_pred_mask)[0]
                
                edge_net_inputs = []
                gt_edges_list = []
                moe_weights_for_edge = []
                
                h_encoded_pos = 0
                for i in range(num_active_graphs):
                    num_nodes_in_graph = (node_roles_gt[active_indices[i], :t - 1] != self.PAD_TOKEN_ID).sum()
                    if active_indices[i] in edge_pred_indices:
                        graph_h = h_encoded[h_encoded_pos : h_encoded_pos + num_nodes_in_graph]
                        num_existing = graph_h.shape[0]

                        if num_existing > 0:
                            new_node_emb = gt_new_node_embeddings[i].unsqueeze(0).repeat(num_existing, 1)
                            task_emb_rep = z[active_indices[i]].unsqueeze(0).repeat(num_existing, 1)
                            
                            edge_net_inputs.append(torch.cat([graph_h, new_node_emb, task_emb_rep], dim=1))
                            gt_edges_list.append(adj_gt[active_indices[i], :num_existing, t - 1])
                            moe_weights_for_edge.append(moe_weights[active_indices[i]].repeat(num_existing, 1))

                    if num_nodes_in_graph > 0:
                        h_encoded_pos += num_nodes_in_graph

                if len(edge_net_inputs) > 0:
                    edge_net_input_batch = torch.cat(edge_net_inputs, dim=0)
                    moe_weights_edge_batch = torch.cat(moe_weights_for_edge, dim=0)
                    gt_edges_batch = torch.cat(gt_edges_list, dim=0)

                    edge_logits = self.moe_edge_net(edge_net_input_batch, moe_weights_edge_batch).squeeze(-1)
                    loss_edge = F.binary_cross_entropy_with_logits(edge_logits, gt_edges_batch, reduction='none')
                    total_loss_graph += loss_edge.sum()

            total_loss_gate_l1 += gate_l1_loss_step * num_active_graphs
        
        num_graphs_in_batch = batch_size
        avg_loss_graph = total_loss_graph / total_node_predictions if total_node_predictions > 0 else 0
        avg_loss_gate_l1 = total_loss_gate_l1 / total_node_predictions if total_node_predictions > 0 else 0
        batch_accuracy = (total_correct_predictions / total_node_predictions) * 100 if total_node_predictions > 0 else 0

        return avg_loss_graph, loss_balance, avg_loss_gate_l1, batch_accuracy

    def sample(self, task_query_embedding, max_nodes=6):
        self.eval()
        device = next(self.parameters()).device
        min_num_nodes = self.args.min_nodes

        with torch.no_grad():
            task_query_embedding = task_query_embedding.to(device)
            z = self.task_encoder(task_query_embedding.unsqueeze(0))

            moe_weights = self.moe_gate(z)

            candidate_ids = list(range(len(self.role_names))) + [self.END_TOKEN_ID]
            candidate_embeddings = self.role_embeddings[candidate_ids]
            end_token_candidate_idx = len(candidate_ids) - 1

            G = nx.DiGraph()
            h_existing = torch.empty(0, self.args.embedding_size_role).to(device)

            t = 0
            while t < max_nodes:
                adj_existing = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32).to(device)
                h_encoded, _ = self.magnet(h_existing, adj_existing, z)

                g_t_minus_1 = self.graph_readout(
                    torch.mean(h_encoded, dim=0) if h_encoded.numel() > 0 else torch.zeros(self.args.hidden_dim,
                                                                                           device=device))

                q_t = self.moe_query_net(g_t_minus_1.unsqueeze(0), moe_weights)
                node_logits = q_t @ candidate_embeddings.t()

                node_probs = F.softmax(node_logits, dim=1)

                if t < min_num_nodes:
                    node_probs[0, end_token_candidate_idx] = 0

                if torch.sum(node_probs).item() < 1e-8:
                    node_probs = torch.ones_like(node_probs)
                    if t < min_num_nodes:
                        node_probs[0, end_token_candidate_idx] = 0
                
                if torch.sum(node_probs).item() > 0:
                    node_probs = node_probs / torch.sum(node_probs)
                else:
                    break

                sampled_candidate_idx = torch.multinomial(node_probs, 1).item()

                new_node_id = candidate_ids[sampled_candidate_idx]

                if new_node_id == self.END_TOKEN_ID:
                    break

                new_node_role = self.id_to_role[new_node_id]
                new_node_embedding = self.role_embeddings[new_node_id]

                G.add_node(t, role=new_node_role)
                h_existing = torch.cat([h_existing, new_node_embedding.unsqueeze(0)], dim=0)

                if G.number_of_nodes() > 1:
                    num_existing = h_encoded.shape[0]
                    edge_net_input = torch.cat([
                        h_encoded,
                        new_node_embedding.unsqueeze(0).repeat(num_existing, 1),
                        z.repeat(num_existing, 1)
                    ], dim=1)

                    edge_logits = self.moe_edge_net(edge_net_input, moe_weights.repeat(num_existing, 1)).squeeze(-1)
                    edge_probs = torch.sigmoid(edge_logits)

                    edge_samples = torch.bernoulli(edge_probs)
                    edges_added = False

                    for i, should_add in enumerate(edge_samples):
                        if should_add.item() == 1:
                            G.add_edge(i, t)
                            edges_added = True

                    if not edges_added:
                        best_edge_idx = torch.argmax(edge_probs).item()
                        G.add_edge(best_edge_idx, t)

                t += 1

            return G
