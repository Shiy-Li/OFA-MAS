import os
import torch
import sys
import random
import networkx as nx
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import from_networkx

# --- Configuration ---
OUTPUT_DIR = './FinetuneData_OFA_Grammar'
NODE_COUNTS = range(3, 7)  # Generate graphs with 3, 4, 5, 6 nodes
SAMPLES_PER_CONFIG = 50  # Number of random role combinations per topology/size
QUESTION_EMBEDDING_SIZE = 384 # all-MiniLM-L6-v2 embedding size

# Ensure the project root is in the Python path to import uni_role
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiment.OFA.uni_role import ROLE_DESCRIPTION

# --- Graph Topology Definitions ---

def create_chain(n):
    """Creates a directed chain graph."""
    G = nx.DiGraph()
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G

def create_star(n):
    """Creates a directed star graph with the first node as the center."""
    G = nx.DiGraph()
    for i in range(1, n):
        G.add_edge(0, i)
    return G

def create_full(n):
    """Creates a fully connected directed graph (excluding self-loops)."""
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j)
    return G

def create_mesh(n):
    """Creates a directed mesh-like graph (grid). For simplicity, we use a 2D grid structure."""
    if n <= 1:
        return nx.DiGraph()
    
    # Find the grid dimensions closest to a square
    rows = int(n**0.5)
    cols = (n + rows - 1) // rows
    
    G = nx.DiGraph()
    node_map = list(range(n))
    
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= n: continue
            
            # Edge to the right
            if c + 1 < cols:
                right_idx = r * cols + (c + 1)
                if right_idx < n:
                    G.add_edge(idx, right_idx)
            
            # Edge downwards
            if r + 1 < rows:
                down_idx = (r + 1) * cols + c
                if down_idx < n:
                    G.add_edge(idx, down_idx)
    return G


TOPOLOGIES = {
    'Chain': create_chain,
    'Star': create_star,
    'FullConnected': create_full,
    'Mesh': create_mesh
}

# --- Main Logic ---

def main():
    """
    Generates a dataset of classic graph topologies with random roles
    for unconditional pre-training of the OFA model.
    """
    print(f"Starting unconditional grammar data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize sentence model and prepare role embeddings
    print("Initializing sentence model and preparing role embeddings...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Use the same caching mechanism for role embeddings
    EMBEDDINGS_CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'precomputed_role_embeddings.pkl')
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print(f"Loading cached role embeddings from {EMBEDDINGS_CACHE_PATH}...")
        with open(EMBEDDINGS_CACHE_PATH, 'rb') as f:
            role_embeddings_dict = pickle.load(f)
    else:
        print("Pre-computing and caching role embeddings...")
        role_embeddings_dict = {
            name: torch.tensor(emb) for name, emb in
            zip(ROLE_DESCRIPTION.keys(), sentence_model.encode(list(ROLE_DESCRIPTION.values())))
        }
        with open(EMBEDDINGS_CACHE_PATH, 'wb') as f:
            pickle.dump(role_embeddings_dict, f)
        print(f"Role embeddings cached to {EMBEDDINGS_CACHE_PATH}")

    all_roles = list(role_embeddings_dict.keys())
    
    # 2. Generate graph data
    print("\nGenerating graph configurations...")
    total_files = len(TOPOLOGIES) * len(NODE_COUNTS) * SAMPLES_PER_CONFIG
    pbar = tqdm(total=total_files, desc="Generating files")
    
    file_counter = 0
    for n_nodes in NODE_COUNTS:
        for topo_name, topo_func in TOPOLOGIES.items():
            for i in range(SAMPLES_PER_CONFIG):
                # a. Randomly sample roles
                if len(all_roles) < n_nodes:
                    print(f"Warning: Not enough unique roles ({len(all_roles)}) to sample {n_nodes}. Skipping.")
                    continue
                
                sampled_roles = random.sample(all_roles, k=n_nodes)
                role_embeddings = [role_embeddings_dict[r] for r in sampled_roles]

                # b. Create graph structure
                G = topo_func(n_nodes)
                
                # Assign roles to nodes for data object
                for idx, role in enumerate(sampled_roles):
                    G.nodes[idx]['role'] = role

                # c. Create PyG Data object
                # First, convert the graph structure. `from_networkx` will also add `role` as a list of strings.
                pyg_data = from_networkx(G)
                
                # Manually add the role embeddings tensor. The data loader expects this attribute name.
                pyg_data.role_embedding = torch.stack(role_embeddings)
                pyg_data.num_nodes = n_nodes

                # d. Add zero vector for unconditional training
                pyg_data.question = torch.zeros(QUESTION_EMBEDDING_SIZE)

                # e. Save file
                filename = f"grammar_{topo_name}_{n_nodes}nodes_sample{i}.pt"
                torch.save(pyg_data, os.path.join(OUTPUT_DIR, filename))
                
                pbar.update(1)
                file_counter += 1

    pbar.close()
    print(f"\n--- Generation Summary ---")
    print(f"Successfully generated {file_counter} graph data files.")
    print(f"Data saved in: {OUTPUT_DIR}")
    print("--------------------------")

if __name__ == '__main__':
    main() 