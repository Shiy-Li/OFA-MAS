import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

# #####################
# # MoE components
# #####################
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
        # x: (batch_size, input_dim)
        # weights: (batch_size, num_experts)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, output_dim)
        # Expand weights to match output dimensions for broadcasting
        expanded_weights = weights.unsqueeze(2)  # (batch_size, num_experts, 1)
        # Weighted sum of expert outputs
        final_output = torch.sum(outputs * expanded_weights, dim=1)  # (batch_size, output_dim)
        return final_output


# #####################
# # TAGSE GNN components
# #####################
class MAGNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, task_dim):
        super(MAGNetLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Context-Gated Message
        self.W_m = nn.Linear(in_dim, out_dim)
        self.W_g = nn.Linear(in_dim + task_dim, out_dim)

        # Role-Aware Attention
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_q = nn.Linear(out_dim, out_dim)
        self.a = nn.Parameter(torch.randn(out_dim * 2, 1))

    def forward(self, h, adj, z):
        # h: node features (num_nodes, in_dim)
        # adj: adjacency matrix (num_nodes, num_nodes)
        # z: task embedding, already expanded for each node (num_nodes, task_dim)
        num_nodes = h.size(0)

        # 1. Context-Gated Message
        raw_message = F.relu(self.W_m(h))
        # The input 'z' is now the per-node expanded task embedding
        gate = torch.sigmoid(self.W_g(torch.cat([h, z], dim=1)))
        gated_message = gate * raw_message  # (num_nodes, out_dim)

        # Calculate L1 loss for gate sparsity (normalized by number of elements)
        gate_l1_loss = torch.norm(gate, p=1) / gate.numel()

        # 2. Role-Aware Attention Aggregation
        q = self.W_q(gated_message)
        k = self.W_k(h)

        q_expanded = q.unsqueeze(1).repeat(1, num_nodes, 1)
        k_expanded = k.unsqueeze(0).repeat(num_nodes, 1, 1)

        attn_input = torch.cat([k_expanded, q_expanded], dim=2)
        e = F.leaky_relu((attn_input @ self.a).squeeze(2))

        # Masking attention scores
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        agg_message = attention.t() @ gated_message

        # 3. State Update
        h_new = (agg_message + h) / 2

        return h_new, gate_l1_loss


class MAGNet(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, task_dim):
        super(MAGNet, self).__init__()
        self.initial_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # All layers will operate on the hidden dimension
            self.layers.append(MAGNetLayer(hidden_dim, hidden_dim, task_dim))

    def forward(self, h, adj, z):
        # Guard against empty input graphs
        if h.shape[0] == 0:
            # For an empty graph, there's no computation and no loss.
            # Must return a tuple of (tensor, loss) to match the non-empty case.
            return h, torch.tensor(0.0, device=z.device)

        # Project initial embeddings into hidden space
        h = self.initial_proj(h)

        # In sampling mode, z might be a single vector that needs to be expanded.
        num_nodes = h.shape[0]
        if num_nodes > 0 and z.shape[0] != num_nodes:
            if z.shape[0] == 1:
                z = z.repeat(num_nodes, 1)
            else:
                # This case should not happen if logic is correct
                raise ValueError("Shape of z is not compatible with h")

        total_gate_l1_loss = 0
        for layer in self.layers:
            h, gate_l1_loss = layer(h, adj, z)
            total_gate_l1_loss += gate_l1_loss
            
        # Return the average gate loss across all layers
        return h, total_gate_l1_loss / len(self.layers)


# #####################
# # OFA Main Model
# #####################
class OFAModel(nn.Module):
    def __init__(self, args, role_embeddings_dict):
        super(OFAModel, self).__init__()
        self.args = args
        # Ensure min_nodes is set, defaulting to 3 if not provided.
        if not hasattr(self.args, 'min_nodes'):
            self.args.min_nodes = 2

        self.task_encoder = nn.Sequential(
            nn.Linear(args.embedding_size_question, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.task_embedding_dim)
        )

        # Role Embeddings
        self.role_names = list(role_embeddings_dict.keys())
        # Register role_embeddings as a buffer to handle device placement automatically
        base_role_embeddings = torch.stack(list(role_embeddings_dict.values()), dim=0)

        # Add START, END, and PAD token embeddings
        start_embedding = torch.zeros(1, self.args.embedding_size_role)
        end_embedding = torch.zeros(1, self.args.embedding_size_role)
        pad_embedding = torch.zeros(1, self.args.embedding_size_role)
        full_embedding_matrix = torch.cat([base_role_embeddings, start_embedding, end_embedding, pad_embedding], dim=0)
        self.register_buffer('role_embeddings', full_embedding_matrix)

        self.id_to_role = {i: name for i, name in enumerate(self.role_names)}
        self.role_to_id = {name: i for i, name in enumerate(self.role_names)}

        # Define START, END, and PAD token IDs
        self.START_TOKEN_ID = len(self.role_names)
        self.END_TOKEN_ID = len(self.role_names) + 1
        self.PAD_TOKEN_ID = len(self.role_names) + 2

        # Add special tokens to the mapping for consistency
        self.id_to_role[self.START_TOKEN_ID] = 'START_TOKEN'
        self.id_to_role[self.END_TOKEN_ID] = 'END_TOKEN'
        self.id_to_role[self.PAD_TOKEN_ID] = 'PAD_TOKEN'
        self.role_to_id['START_TOKEN'] = self.START_TOKEN_ID
        self.role_to_id['END_TOKEN'] = self.END_TOKEN_ID
        self.role_to_id['PAD_TOKEN'] = self.PAD_TOKEN_ID

        # MoE Gating Network
        self.moe_gate = nn.Sequential(
            nn.Linear(args.task_embedding_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.num_experts),
            nn.Softmax(dim=-1)
        )

        # MAGNet GNN
        self.magnet = MAGNet(
            num_layers=args.gcn_layers,
            in_dim=args.embedding_size_role,
            hidden_dim=args.hidden_dim,
            task_dim=args.task_embedding_dim
        )

        # Readout function to get global graph vector
        self.graph_readout = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU()
        )

        # MoE Query Network for new node prediction
        self.moe_query_net = MoELayer(
            num_experts=args.num_experts,
            input_dim=args.hidden_dim,
            output_dim=args.embedding_size_role
        )
        # MoE Edge Prediction Network
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

        if self.args.use_vae:
            recon_task, z, mu, logvar = self.task_vae(task_embedding)
            recon_loss = F.mse_loss(recon_task, task_embedding)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            loss_vae = torch.mean(recon_loss + beta_vae * kld_loss)
        else:
            z = self.task_encoder(task_embedding)
            loss_vae = torch.tensor(0.0, device=device)

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

        # Create a mask for active graphs at each timestep
        t_range = torch.arange(1, max_nodes + 1, device=device).unsqueeze(0)
        active_mask = (t_range <= graph_sizes.unsqueeze(1)).float()

        for t in range(1, max_nodes):
            num_active_graphs = int(active_mask[:, t].sum())
            if num_active_graphs == 0:
                continue

            active_indices = torch.where(active_mask[:, t] == 1)[0]
            
            # --- Prepare batched input for MAGNet ---
            # Create a block diagonal adjacency matrix for all active graphs at step t
            sub_adjs = adj_gt[active_indices, :t - 1, :t - 1]
            batched_adj = torch.block_diag(*sub_adjs)
            
            # Get node embeddings
            sub_role_ids = node_roles_gt[active_indices, :t - 1].reshape(-1)
            # Filter out padding
            valid_node_indices = torch.where(sub_role_ids != self.PAD_TOKEN_ID)[0]
            
            h_existing = self.role_embeddings[sub_role_ids[valid_node_indices]]
            
            # Filter batched_adj as well
            batched_adj = batched_adj[valid_node_indices][:, valid_node_indices]

            # Prepare task embeddings for MAGNet (needs to be repeated for each node)
            z_active = z[active_indices]
            node_to_graph_map = torch.arange(num_active_graphs, device=device).repeat_interleave(t - 1)
            z_repeated = z_active[node_to_graph_map[valid_node_indices]]

            # Run MAGNet
            h_encoded, gate_l1_loss_step = self.magnet(h_existing, batched_adj, z_repeated)
            
            # --- Graph Readout ---
            # We need to aggregate node embeddings graph by graph
            g_t_minus_1_list = []
            current_pos = 0
            for i in range(num_active_graphs):
                # Number of valid (non-padded) nodes in this graph at step t-1
                num_nodes_in_graph = (node_roles_gt[active_indices[i], :t - 1] != self.PAD_TOKEN_ID).sum()
                if num_nodes_in_graph > 0:
                    graph_h = h_encoded[current_pos : current_pos + num_nodes_in_graph]
                    g_t_minus_1_list.append(self.graph_readout(torch.mean(graph_h, dim=0)))
                    current_pos += num_nodes_in_graph
                else:
                    g_t_minus_1_list.append(torch.zeros(self.args.hidden_dim, device=device))
            
            g_t_minus_1 = torch.stack(g_t_minus_1_list)

            # --- Node Prediction Loss ---
            q_t = self.moe_query_net(g_t_minus_1, moe_weights[active_indices])
            node_logits = q_t @ self.role_embeddings.t()
            gt_node_ids = node_roles_gt[active_indices, t - 1]

            loss_node = F.cross_entropy(node_logits, gt_node_ids, reduction='none')
            total_loss_graph += loss_node.sum()

            # Accuracy
            pred_node_ids = torch.argmax(node_logits, dim=1)
            total_correct_predictions += (pred_node_ids == gt_node_ids).sum().item()
            total_node_predictions += num_active_graphs

            # --- Edge Prediction Loss ---
            gt_new_node_embeddings = self.role_embeddings[gt_node_ids]
            # No edges to END or PAD tokens
            valid_edge_pred_mask = (gt_node_ids != self.END_TOKEN_ID) & (gt_node_ids != self.PAD_TOKEN_ID)
            
            if t > 1 and valid_edge_pred_mask.any():
                edge_pred_indices = torch.where(valid_edge_pred_mask)[0]
                
                # We need to construct a batch for edge prediction for graphs that need it
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

        return avg_loss_graph, loss_vae, loss_balance, avg_loss_gate_l1, batch_accuracy

    def sample(self, task_query_embedding, max_nodes=6):
        """
        Generates a graph autoregressively for a given task query embedding.
        """
        self.eval()
        device = next(self.parameters()).device
        min_num_nodes = self.args.min_nodes

        with torch.no_grad():
            task_query_embedding = task_query_embedding.to(device)
            # Unsqueeze to add a batch dimension of 1
            z = self.task_encoder(task_query_embedding.unsqueeze(0))

            moe_weights = self.moe_gate(z)

            # Create a candidate vocabulary for prediction that excludes START_TOKEN
            candidate_ids = list(range(len(self.role_names))) + [self.END_TOKEN_ID]
            candidate_embeddings = self.role_embeddings[candidate_ids]
            end_token_candidate_idx = len(candidate_ids) - 1

            # 2. Initialize graph
            G = nx.DiGraph()
            h_existing = torch.empty(0, self.args.embedding_size_role).to(device)

            t = 0
            while t < max_nodes:
                # 3. Encode current graph state
                adj_existing = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32).to(device)
                # We don't need the gate loss during sampling
                h_encoded, _ = self.magnet(h_existing, adj_existing, z)

                g_t_minus_1 = self.graph_readout(
                    torch.mean(h_encoded, dim=0) if h_encoded.numel() > 0 else torch.zeros(self.args.hidden_dim,
                                                                                           device=device))

                # 4. Predict new node against the candidate vocabulary
                q_t = self.moe_query_net(g_t_minus_1.unsqueeze(0), moe_weights)
                node_logits = q_t @ candidate_embeddings.t()

                node_probs = F.softmax(node_logits, dim=1)

                # Prevent sampling END_TOKEN if graph is too small
                if t < min_num_nodes:
                    node_probs[0, end_token_candidate_idx] = 0

                # Handle cases where all valid probabilities are near zero
                if torch.sum(node_probs).item() < 1e-8:
                    # Reset to a uniform distribution
                    node_probs = torch.ones_like(node_probs)
                    # Re-apply the mask to ensure min_nodes is respected
                    if t < min_num_nodes:
                        node_probs[0, end_token_candidate_idx] = 0
                
                # Final normalization to ensure probabilities sum to 1
                if torch.sum(node_probs).item() > 0:
                    node_probs = node_probs / torch.sum(node_probs)
                else:
                    # Fallback if all probabilities are zero after masking, break gracefully
                    break

                sampled_candidate_idx = torch.multinomial(node_probs, 1).item()

                # Map the candidate index back to the original, global token ID
                new_node_id = candidate_ids[sampled_candidate_idx]

                # Check for END_TOKEN to stop generation
                if new_node_id == self.END_TOKEN_ID:
                    break

                new_node_role = self.id_to_role[new_node_id]
                new_node_embedding = self.role_embeddings[new_node_id]

                # Add node to graph
                G.add_node(t, role=new_node_role)
                h_existing = torch.cat([h_existing, new_node_embedding.unsqueeze(0)], dim=0)

                # 5. Predict edges to the new node
                if G.number_of_nodes() > 1:
                    num_existing = h_encoded.shape[0]
                    edge_net_input = torch.cat([
                        h_encoded,
                        new_node_embedding.unsqueeze(0).repeat(num_existing, 1),
                        z.repeat(num_existing, 1)
                    ], dim=1)

                    edge_logits = self.moe_edge_net(edge_net_input, moe_weights.repeat(num_existing, 1)).squeeze(-1)
                    edge_probs = torch.sigmoid(edge_logits)

                    # Strategy 1: Use probabilistic sampling (like ARG-Designer)
                    edge_samples = torch.bernoulli(edge_probs)
                    edges_added = False

                    for i, should_add in enumerate(edge_samples):
                        if should_add.item() == 1:
                            G.add_edge(i, t)
                            edges_added = True

                    # Strategy 2: Ensure at least one edge (inspired by ARG-Designer)
                    if not edges_added:
                        # If no edges were sampled, force connect to the node with highest probability
                        best_edge_idx = torch.argmax(edge_probs).item()
                        G.add_edge(best_edge_idx, t)

                t += 1  # Increment node counter

            return G
