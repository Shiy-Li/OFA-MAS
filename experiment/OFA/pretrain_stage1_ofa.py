import torch
import random
import numpy as np

from experiment.OFA.process_dataset import load_graph_dataset
from experiment.OFA.args_ofa import pretrain_stage1_args
from experiment.OFA.model_ofa import OFAModel
from experiment.OFA.train_ofa import train


def main():
    args = pretrain_stage1_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading dataset for Stage 1: Unconditional Pre-training...")
    train_dataset, role_embeddings_dict = load_graph_dataset(args)

    if len(train_dataset) == 0:
        print(f"Dataset is empty. Looked in: {args.data_dir}. Please generate grammar data first. Exiting.")
        return

    print(f"Dataset loaded with {len(train_dataset)} graphs.")

    print("Initializing model for Stage 1...")
    model = OFAModel(args, role_embeddings_dict)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    print("Starting Stage 1: Unconditional Pre-training...")
    print('args.lambda_gate_l1', args.lambda_gate_l1)
    trained_model = train(args, train_dataset, model, is_pretraining=True, unconditional=True)


if __name__ == '__main__':
    main()
