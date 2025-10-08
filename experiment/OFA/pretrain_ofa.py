import torch
import os
import random
import numpy as np

from experiment.OFA.process_dataset import load_graph_dataset
from experiment.OFA.args_ofa import pretrain_stage2_args
from experiment.OFA.model_ofa import OFAModel
from experiment.OFA.train_ofa import train

def main():
    args = pretrain_stage2_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading dataset for Stage 2: Conditional Pre-training...")
    train_dataset, role_embeddings_dict = load_graph_dataset(args)
    
    if len(train_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    print(f"Dataset loaded with {len(train_dataset)} graphs.")
    
    print("Initializing model...")
    print('role_embeddings_dict', role_embeddings_dict.keys())
    model = OFAModel(args, role_embeddings_dict)

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    model = model.to(device)

    if args.stage1_model_path and os.path.exists(args.stage1_model_path):
        print(f"Loading Stage 1 pre-trained model from {args.stage1_model_path}...")
        try:
            checkpoint = torch.load(args.stage1_model_path, map_location=device)
            if 'role_embeddings' in checkpoint:
                checkpoint.pop('role_embeddings')
            model.load_state_dict(checkpoint, strict=False)
            print("Stage 1 model loaded successfully.")
        except Exception as e:
            print(f"Could not load Stage 1 model: {e}. Starting from scratch for Stage 2.")
    else:
        print("No Stage 1 model path provided or file not found. Starting Stage 2 from scratch.")


    print("Starting Stage 2: Conditional Pre-training...")
    trained_model = train(args, train_dataset, model, is_pretraining=True)
    
if __name__ == '__main__':
    main()
 