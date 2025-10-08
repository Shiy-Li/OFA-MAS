import os
import json
import argparse
from tqdm import tqdm
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from sentence_transformers import SentenceTransformer
import re
import pickle

from experiment.OFA.uni_role import ROLE_DESCRIPTION

def pre_compute_role_embeddings(model_name='all-MiniLM-L6-v2', cache_path='precomputed_role_embeddings.pkl'):
    if os.path.exists(cache_path):
        print(f"Loading pre-computed role embeddings from {cache_path}...")
        with open(cache_path, 'rb') as f:
            role_embeddings = pickle.load(f)
        return role_embeddings

    print("Pre-computing role embeddings...")
    model = SentenceTransformer(model_name)
    role_names = list(ROLE_DESCRIPTION.keys())
    role_descriptions = list(ROLE_DESCRIPTION.values())
    embeddings = model.encode(role_descriptions, show_progress_bar=True)
    
    role_embeddings = {name: torch.tensor(emb) for name, emb in zip(role_names, embeddings)}
    
    print(f"Saving role embeddings to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(role_embeddings, f)
        
    return role_embeddings

def process_llm_data(input_file, output_dir, role_embeddings, sentence_model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} entries from {input_file}...")
    invalid_escape_re = re.compile(r'\\([^"\\/bfnrtu])')

    for i, line in enumerate(tqdm(lines)):
        try:
            line_corrected = invalid_escape_re.sub(r'\1', line)
            item = json.loads(line_corrected.strip())
            
            G = nx.DiGraph()
            
            for node_idx, role_name in enumerate(item['roles']):
                if role_name in role_embeddings:
                    G.add_node(node_idx, role=role_name, role_embedding=role_embeddings[role_name])
                else:
                    print(f"Warning: Role '{role_name}' not found in pre-computed embeddings. Skipping.")
                    continue

            if 'edges' in item and item['edges']:
                edge_list = item['edges'].split()
                for edge_str in edge_list:
                    if '->' in edge_str:
                        u, v = map(int, edge_str.split('->'))
                        G.add_edge(u, v)
            
            G.graph['question'] = item['query']
            G.graph['topology'] = item['topology']
            pyg_data = from_networkx(G)

            pyg_data.question = sentence_model.encode(item['query'], convert_to_tensor=True)
            
            torch.save(pyg_data, os.path.join(output_dir, f"graph_{i}.pt"))

        except json.JSONDecodeError as e:
            print(f"Warning: Skipping invalid JSON on line {i+1}: {e}")
        except Exception as e:
            print(f"An error occurred on line {i+1}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process LLM-generated data for OFA pre-training.")
    parser.add_argument('--input_file', type=str, default='./llm_generated_ofa_data.jsonl', help='Path to the input JSONL file.')
    parser.add_argument('--output_dir', type=str, default='../PTData/', help='Path to the directory to save processed graph files.')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Name of the sentence-transformer model to use.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist to store the cache file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    embedding_cache_path = os.path.join(args.output_dir, 'precomputed_role_embeddings.pkl')
    role_embeddings = pre_compute_role_embeddings(args.model_name, cache_path=embedding_cache_path)
    sentence_model = SentenceTransformer(args.model_name)
    
    process_llm_data(args.input_file, args.output_dir, role_embeddings, sentence_model)
    
    print("Data processing complete.")

if __name__ == '__main__':
    main() 