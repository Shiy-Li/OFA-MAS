import torch
import os
import random
import numpy as np
import copy
from torch.utils.data import ConcatDataset

from experiment.OFA.process_dataset import load_graph_dataset
from experiment.OFA.args_ofa import finetune_args
from experiment.OFA.model_ofa import OFAModel
from experiment.OFA.train_ofa import train
from sentence_transformers import SentenceTransformer
from experiment.OFA.uni_role import ROLE_DESCRIPTION

def main():
    args = finetune_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Load datasets from multiple directories and combine them
    print("Loading fine-tuning datasets from multiple sources...")
    all_datasets = []
    args_clone = copy.deepcopy(args)

    for data_dir in args.data_dirs:
        # FIX: Check if the directory exists before trying to load from it
        if os.path.exists(data_dir):
            print(f" -> Loading from {data_dir}")
            args_clone.data_dir = data_dir  # Set the current directory for the loader
            dataset, _ = load_graph_dataset(args_clone)
            if len(dataset) > 0:
                all_datasets.append(dataset)
                print(f"    Loaded {len(dataset)} graphs.")
        else:
            print(f" -> Warning: Directory not found, skipping: {data_dir}")
    
    if not all_datasets:
        print("No data found in any of the specified directories. Exiting.")
        return

    train_dataset = ConcatDataset(all_datasets)
    print(f"Combined dataset loaded with {len(train_dataset)} successfully executed graphs.")
    
    # Initialize model
    print("Initializing model...")
    # We need role_embeddings to initialize the model, even if we load a checkpoint
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # The role embeddings are generated directly from the ROLE_DESCRIPTION.
    # The OFAModel is responsible for adding START and END tokens internally.
    role_embeddings_dict = {
        name: torch.tensor(emb) for name, emb in 
        zip(ROLE_DESCRIPTION.keys(), sentence_model.encode(list(ROLE_DESCRIPTION.values())))
    }
    model = OFAModel(args, role_embeddings_dict)

    # Set device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Load pre-trained model weights from Stage 2
    print(f"Loading Stage 2 pre-trained model from {args.stage2_model_path}...")
    try:
        checkpoint = torch.load(args.stage2_model_path, map_location=device)
        
        # Handle the role embedding mismatch due to vocabulary change
        # by manually copying the compatible weights.
        if 'role_embeddings' in checkpoint:
            pretrained_embeddings = checkpoint.pop('role_embeddings')
            current_embeddings_param = model.state_dict()['role_embeddings']
            
            # Get sizes
            n_old_plus_specials, emb_dim = pretrained_embeddings.shape
            n_new_plus_specials, _ = current_embeddings_param.shape
            n_old = n_old_plus_specials - 2 # Number of base roles in pretrained model
            
            # Copy the weights for the common base roles
            # This assumes new roles were added to the end of the dictionary
            common_roles_count = min(n_old, n_new_plus_specials - 2)
            current_embeddings_param.data[:common_roles_count, :] = pretrained_embeddings.data[:common_roles_count, :]
            
            # Copy the START/END token embeddings from the end of the pretrained tensor
            current_embeddings_param.data[-2:, :] = pretrained_embeddings.data[-2:, :]
            
            print(f"Successfully loaded and adapted role_embeddings: {n_old} base roles from checkpoint, {current_embeddings_param.shape[0]-2} in current model.")

            model.load_state_dict(checkpoint, strict=False)

        print("Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"Could not load pre-trained model: {e}. Starting from scratch.")

    # Start fine-tuning
    print("Starting fine-tuning...")
    trained_model = train(args, train_dataset, model)
    
if __name__ == '__main__':
    main()
