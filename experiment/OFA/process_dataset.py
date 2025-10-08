import torch.utils.data


def load_graph_dataset(args):
    if args.dataset in ['ofa_stage1_unconditional', 'ofa_stage2_conditional', 'ofa_finetune_combined']:
        from experiment.OFA.ofa_adapter import OFAGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        dataset = OFAGraphDataset(data_dir=args.data_dir, sample_size=sample_size)

        args.max_prev_node = 3
        args.max_head_and_tail = None
        
        role_embeddings_dict = {}
        if dataset.graph_list:
            for graph in dataset.graph_list:
                for _, data in graph.nodes(data=True):
                    if 'role' in data and 'role_embedding' in data:
                        embedding = data['role_embedding']
                        if not isinstance(embedding, torch.Tensor):
                            embedding = torch.tensor(embedding)
                        role_embeddings_dict[data['role']] = embedding
        return dataset, role_embeddings_dict

    else:
        raise Exception(f"Unsupported dataset: {args.dataset}")