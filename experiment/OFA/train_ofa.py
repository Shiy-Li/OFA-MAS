import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import networkx as nx
import os

def collate_for_ofa(batch_graphs, device, role_to_id, pad_token_id):
    if not batch_graphs:
        return {}
    max_nodes = 0
    for g in batch_graphs:
        max_nodes = max(max_nodes, len(g.graph['final_roles']))

    batch_size = len(batch_graphs)
    embedding_dim = batch_graphs[0].graph['task_embedding'].shape[0]
    task_embeddings = torch.zeros(batch_size, embedding_dim, device=device)
    adj_gt = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    node_roles = torch.full((batch_size, max_nodes), pad_token_id, dtype=torch.long, device=device)
    graph_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, g in enumerate(batch_graphs):
        roles = g.graph['final_roles']
        num_nodes = len(roles)
        
        task_embeddings[i] = g.graph['task_embedding'].to(device)
        graph_sizes[i] = num_nodes
        
        role_ids = torch.tensor([role_to_id[r] for r in roles], dtype=torch.long, device=device)
        node_roles[i, :num_nodes] = role_ids
        
        num_real_nodes = num_nodes - 1
        if num_real_nodes > 0:
            nodes = sorted(list(g.nodes()))
            original_adj = torch.tensor(nx.to_numpy_array(g, nodelist=nodes), dtype=torch.float32, device=device)
            adj_gt[i, :num_real_nodes, :num_real_nodes] = original_adj
            
    return {
        'task_embedding': task_embeddings,
        'adj_gt': adj_gt,
        'node_roles': node_roles,
        'graph_sizes': graph_sizes
    }

def train(args, train_dataset, model, is_pretraining=False, unconditional=False):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay_factor, patience=args.lr_decay_patience, verbose=True)
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    collate_fn_with_args = lambda batch: collate_for_ofa(batch, device, model.role_to_id, model.PAD_TOKEN_ID)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_args)

    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss_all = 0
        total_loss_graph = 0
        total_loss_balance = 0
        total_loss_gate_l1 = 0
        total_role_accuracy = 0
        
        beta_vae = min(1.0, epoch / args.beta_vae_anneal_epochs)

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            loss_graph, loss_balance, loss_gate_l1, accuracy = model(batch, beta_vae, unconditional=unconditional)
            
            loss = args.lambda_graph * loss_graph + \
                   args.lambda_balance * loss_balance + \
                   args.lambda_gate_l1 * loss_gate_l1
            
            loss.backward()
            optimizer.step()
            
            total_loss_all += loss.item()
            total_loss_graph += loss_graph.item()
            total_loss_balance += loss_balance.item()
            total_loss_gate_l1 += loss_gate_l1.item()
            total_role_accuracy += accuracy

        avg_loss = total_loss_all / len(dataloader)
        avg_loss_graph = total_loss_graph / len(dataloader)
        avg_loss_balance = total_loss_balance / len(dataloader)
        avg_loss_gate_l1 = total_loss_gate_l1 / len(dataloader)
        avg_role_accuracy = total_role_accuracy / len(dataloader)

        scheduler.step(avg_loss)
        
        end_time = time.time()
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {end_time - start_time:.2f}s")
        print(f"  Avg Total Loss: {avg_loss:.4f} | Avg Role Accuracy: {avg_role_accuracy:.2f}%")
        print(f"  Losses -> Graph: {avg_loss_graph:.4f}, Balance: {avg_loss_balance:.4f}, GateL1: {avg_loss_gate_l1:.4f}")

        if (epoch + 1) % 100 == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            
            if unconditional:
                model_path = os.path.join(args.save_dir, f'ofa_stage1_unconditional_{args.lr}_{epoch+1}_sparse.pth')
            elif is_pretraining:
                model_path = os.path.join(args.save_dir, f'ofa_stage2_conditional_{args.lr}_{epoch+1}_sparse.pth')
            else:
                dataset_names = [os.path.basename(d.strip('/')).replace('Finetune_OFA', '') for d in args.data_dirs]
                dataset_str = '_'.join(dataset_names)
                model_path = os.path.join(args.save_dir, f'ofa_finetune_{dataset_str}_{args.lr}_{epoch+1}_sparse.pth')

            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} at epoch {epoch+1}")
        
    return model
