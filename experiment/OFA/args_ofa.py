import argparse

def base_args():

    parser = argparse.ArgumentParser(description='OFA Base Arguments', add_help=False)

    parser.add_argument('--data_dir', type=str, default='../PTData',
                        help='Directory containing the pre-training graph data.')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save the trained models.')
    parser.add_argument('--dataset', type=str, default='ofa_pretrain',
                        help='Dataset name. (used as default)')
    parser.add_argument('--dataset_sample_size', type=int, default=0,
                        help='Number of samples to use from the dataset (0 for all).')

    parser.add_argument('--epochs', type=int, default=101,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='Learning rate decay factor.')
    parser.add_argument('--lr_decay_patience', type=int, default=10,
                        help='Patience for learning rate decay.')

    parser.add_argument('--embedding_size_question', type=int, default=384,
                        help='Embedding size of the question/task from sentence transformer.')
    parser.add_argument('--embedding_size_role', type=int, default=384,
                        help='Embedding size of the role description.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size for GNN and MLPs.')

    parser.add_argument('--task_embedding_dim', type=int, default=128,
                        help='Latent dimension size for the Task Encoder.')

    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of experts in the Mixture-of-Experts layers.')

    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='Number of layers in the MAGNet GNN.')

    parser.add_argument('--lambda_graph', type=float, default=1.0, help='Weight for graph generation loss.')
    parser.add_argument('--lambda_balance', type=float, default=0.2, help='Weight for MoE balance loss.')
    parser.add_argument('--lambda_gate_l1', type=float, default=0.1, help='Weight for MAGNet gate L1 sparsity loss.')
    parser.add_argument('--beta_vae_anneal_epochs', type=int, default=50,
                        help='Number of epochs to anneal the VAE KL divergence beta weight from 0 to 1.')

    parser.add_argument('--use_gpu', dest='use_gpu', default=True,
                        help='Enable GPU usage (default True). Use --no-use_gpu to disable.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_nodes', type=int, default=6, help='Maximum number of nodes for graph generation during sampling (acts as a safeguard).')
    parser.add_argument('--min_nodes', type=int, default=2, help='Minimum number of nodes to generate before allowing END token.')

    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Name of the sentence-transformer model to use.')

    args, _ = parser.parse_known_args()
    return parser, args


def pretrain_stage1_args():
    parent_parser, _ = base_args()
    parser = argparse.ArgumentParser(description='OFA Stage 1: Unconditional Pre-training',
                                     parents=[parent_parser],
                                     conflict_handler='resolve')
    parser.set_defaults(
        dataset='ofa_stage1_unconditional',
        epochs=51,
        lr=2e-3,
        save_dir='./saved_models_stage1/'
    )
    args = parser.parse_args()
    return args


def pretrain_stage2_args():
    parent_parser, _ = base_args()
    parser = argparse.ArgumentParser(description='OFA Stage 2: Conditional Pre-training',
                                     parents=[parent_parser],
                                     conflict_handler='resolve')
    parser.add_argument('--stage1_model_path', type=str,
                        default=f'./saved_models_stage1/xxxx.pth',
                        help='Path to the stage 1 pre-trained model checkpoint.')
    parser.set_defaults(
        dataset='ofa_stage2_conditional',
        epochs=51,
        lr=2e-3,
        save_dir='./saved_models_stage2/'
    )
    args = parser.parse_args()
    return args

def finetune_stage3_args():
    pretrain_parser, defaults = base_args()

    parser = argparse.ArgumentParser(description='OFA Fine-tuning Arguments',
                                     parents=[pretrain_parser],
                                     conflict_handler='resolve')

    parser.add_argument('--dataset', type=str, default='ofa_finetune_combined', help='A name for the combined fine-tuning dataset.')
    parser.add_argument('--data_dirs', nargs='+',
                        default=[
                            './FinetuneData_Generation/Finetune_OFA_mmlu/',
                            './FinetuneData_Generation/Finetune_OFA_humaneval/',
                            './FinetuneData_Generation/Finetune_OFA_gsm8k/',
                            './FinetuneData_Generation/Finetune_OFA_aqua/',
                            './FinetuneData_Generation/Finetune_OFA_svamp/',
                            './FinetuneData_Generation/Finetune_OFA_multiarith/'
                        ],
                        help='A list of directories containing the fine-tuning data.')
    parser.add_argument('--stage2_model_path', type=str, default='./saved_models_stage2/xxx.pth')

    parser.set_defaults(
        save_dir='./saved_models_stage3/',
        epochs=51,
        lr=5e-4
    )

    args = parser.parse_args()
    return args


def evaluate_args():
    parser = argparse.ArgumentParser(description='OFA Evaluation Arguments')

    _, defaults = base_args()

    parser.add_argument('--dataset', nargs='+', type=str,
                        default=['mmlu'],
                        help='Dataset(s) to evaluate on.')
    parser.add_argument('--model_path', type=str,
                        default='xxx',
                        help='Path to the pre-trained or fine-tuned OFA model.')
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM to use for MAS execution.")
    parser.add_argument('--limit', type=int, default=448, help="Limit number of evaluation samples.")
    parser.add_argument('--eval_batch_size', type=int, default=64, help="Batch size for parallel evaluation.")
    parser.add_argument('--summary_log_file', type=str, default='xxx/evaluation_summary.jsonl',
                        help="Log file to record evaluation summaries.")

    parser.add_argument('--model_name', type=str, default=defaults.model_name, help='Name of the sentence-transformer model to use.')
    parser.add_argument('--max_nodes', type=int, default=6, help='Maximum number of nodes in the generated graph.')
    parser.add_argument('--min_nodes', type=int, default=2, help='Minimum number of nodes to generate before allowing END token.')
    parser.add_argument('--greedy', default=False, help='Use sampling instead of greedy decoding for graph generation.')

    parser.add_argument('--embedding_size_question', type=int, default=defaults.embedding_size_question)
    parser.add_argument('--task_embedding_dim', type=int, default=defaults.task_embedding_dim)
    parser.add_argument('--embedding_size_role', type=int, default=defaults.embedding_size_role)
    parser.add_argument('--hidden_dim', type=int, default=defaults.hidden_dim)
    parser.add_argument('--num_experts', type=int, default=defaults.num_experts)
    parser.add_argument('--gcn_layers', type=int, default=defaults.gcn_layers)
    parser.add_argument('--use_vae', help='Use TaskVAE for task representation.', default=defaults.use_vae)
    parser.add_argument('--use_gpu', default=defaults.use_gpu)
    parser.add_argument('--seed', default=defaults.seed)

    args = parser.parse_args()
    return args

pretrain_args = pretrain_stage2_args
finetune_args = finetune_stage3_args
