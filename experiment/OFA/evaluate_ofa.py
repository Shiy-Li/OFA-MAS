import os
import json
import math
import asyncio
import time
import torch
import numpy as np
import random
import sys
import datetime
import copy
from tqdm import tqdm
from typing import List, Any, Iterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from sentence_transformers import SentenceTransformer
from GDesigner.graph.graph import TestGraph
from GDesigner.tools.reader.readers import JSONLReader, JSONReader
from GDesigner.tools.coding.python_executor import PyExecutor

from experiment.OFA.args_ofa import evaluate_args
from experiment.OFA.model_ofa import OFAModel
from experiment.OFA.uni_role import ROLE_DESCRIPTION

from datasets.mmlu_dataset import MMLUDataset
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict, svamp_data_process, multiarith_data_process
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict


def convert_to_pyg_graph(nx_graph, task_text):
    from torch_geometric.data import Data
    pyg = Data()
    num_nodes = nx_graph.number_of_nodes()
    features = []
    for i in range(num_nodes):
        features.append({
            'role': nx_graph.nodes[i].get('role', 'Unknown'),
            'constraint': nx_graph.nodes[i].get('constraint', ROLE_DESCRIPTION[nx_graph.nodes[i].get('role', 'Unknown')])
        })
    pyg.x = features
    edges = [[u, v] for u, v in nx_graph.edges()]
    pyg.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
    pyg.task = task_text
    pyg.num_nodes = num_nodes
    return pyg

def get_dataset_config(dataset_name):
    print(f"Loading and configuring for dataset: {dataset_name}")
    
    if dataset_name == 'mmlu':
        from datasets.MMLU.download import download
        download()
        dataset = MMLUDataset('val')
        dataset_specific_info = {
            'record_to_input': dataset.record_to_input,
            'postprocess_answer': dataset.postprocess_answer,
            'check_correctness': lambda pred, rec: pred == dataset.record_to_target_answer(rec),
            'decision_method': "FinalRefer"
        }
    elif dataset_name == 'gsm8k':
        full_dataset_raw = JSONLReader.parse_file('../../datasets/gsm8k/gsm8k.jsonl')
        full_dataset = gsm_data_process(full_dataset_raw)
        with open('./FinetuneData_Generation/task_split_gsm8k.json', 'r') as f:
            test_indices = json.load(f)['test_indices']
        dataset = [full_dataset[i] for i in test_indices]
        dataset_specific_info = {
            'record_to_input': lambda rec: {"task": rec["task"]},
            'postprocess_answer': lambda ans: gsm_get_predict(ans),
            'check_correctness': lambda pred, rec: float(pred) == float(rec["answer"]) if pred is not None and pred != '' else False,
            'decision_method': "FinalRefer",
            'domain': 'gsm8k'
        }
    elif dataset_name == 'humaneval':
        full_dataset = list(JSONLReader.parse_file('../../datasets/humaneval/humaneval-py.jsonl'))
        executor = PyExecutor()
        with open('./FinetuneData_Generation/task_split_humaneval.json', 'r') as f:
            test_indices = json.load(f)['test_indices']
        dataset = [full_dataset[i] for i in test_indices]
        dataset_specific_info = {
            'record_to_input': lambda rec: {"task": rec["prompt"]},
            'postprocess_answer': lambda ans: ans.lstrip("```python\n").rstrip("\n```"),
            'check_correctness': lambda pred, rec: executor.execute(pred, [rec["test"]], timeout=100)[0],
            'decision_method': "FinalWriteCode",
        }
    elif dataset_name == 'svamp':
        full_dataset_raw = JSONReader.parse_file('../../datasets/SVAMP/SVAMP.json')
        full_dataset = svamp_data_process(full_dataset_raw)
        split_file = './FinetuneData_Generation/task_split_svamp.json'
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"SVAMP task split file not found at {split_file}. "
                f"Please run generate_finetune_data_svamp.py first."
            )
        with open(split_file, 'r') as f:
            test_indices = json.load(f)['test_indices']
        dataset = [full_dataset[i] for i in test_indices]
        dataset_specific_info = {
            'record_to_input': lambda rec: {"task": rec["task"]},
            'postprocess_answer': lambda ans: gsm_get_predict(ans),
            'check_correctness': lambda pred, rec: float(pred) == float(rec["answer"]) if pred is not None and pred != '' else False,
            'decision_method': "FinalRefer",
            'domain': 'gsm8k'
        }
    elif dataset_name == 'aqua':
        full_dataset_raw = JSONLReader.parse_file('../../datasets/AQuA/AQuA.jsonl')
        full_dataset = aqua_data_process(full_dataset_raw)
        split_file = './FinetuneData_Generation/task_split_aqua.json'
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"AQuA task split file not found at {split_file}. "
                f"Please run generate_finetune_data_aqua.py first."
            )
        with open(split_file, 'r') as f:
            test_indices = json.load(f)['test_indices']
        dataset = [full_dataset[i] for i in test_indices]
        dataset_specific_info = {
            'record_to_input': lambda rec: {"task": rec["task"]},
            'postprocess_answer': lambda ans: aqua_get_predict(ans),
            'check_correctness': lambda pred, rec: pred == rec["answer"],
            'decision_method': "FinalRefer"
        }
    elif dataset_name == 'multiarith':
        full_dataset_raw = JSONReader.parse_file('../../datasets/MultiArith/MultiArith.json')
        full_dataset = multiarith_data_process(full_dataset_raw)
        split_file = './FinetuneData_Generation/task_split_multiarith.json'
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"MultiArith task split file not found at {split_file}. "
                f"Please run generate_finetune_data_multiarith.py first."
            )
        with open(split_file, 'r') as f:
            test_indices = json.load(f)['test_indices']
        dataset = [full_dataset[i] for i in test_indices]
        dataset_specific_info = {
            'record_to_input': lambda rec: {"task": rec["task"]},
            'postprocess_answer': lambda ans: gsm_get_predict(ans),
            'check_correctness': lambda pred, rec: float(pred) == float(rec["answer"]) if pred is not None and pred != '' else False,
            'decision_method': "FinalRefer",
            'domain': 'gsm8k'
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return dataset, dataset_specific_info


async def evaluate(args, model, dataset, sentence_model, dataset_specific_info):
    print(f"Starting evaluation on {args.dataset}...")
    
    record_to_input = dataset_specific_info['record_to_input']
    postprocess_answer = dataset_specific_info['postprocess_answer']
    check_correctness = dataset_specific_info['check_correctness']
    decision_method = dataset_specific_info['decision_method']
    domain = dataset_specific_info.get('domain', args.dataset)

    total_correct = 0
    results_list = []
    if args.dataset == 'mmlu':
        args.limit = 153
    elif args.dataset == 'gsm8k':
        args.limit = 1000

    def eval_loader(data, batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i, record in enumerate(data):
            if args.limit is not None and i >= args.limit:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records
    
    data_len = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    num_batches = int(math.ceil(data_len / args.eval_batch_size))

    pbar = tqdm(enumerate(eval_loader(dataset, args.eval_batch_size)), total=num_batches, desc=f"Evaluating on {args.dataset}", ncols=120)
    
    for i_batch, record_batch in pbar:
        answer_tasks = []
        metadata_for_tasks = []

        for record in record_batch:
            input_dict = record_to_input(record)
            task_text = input_dict['task']
            
            query_embedding = sentence_model.encode(task_text, convert_to_tensor=True).to(model.role_embeddings.device)
            generated_graph = model.sample(
                task_query_embedding=query_embedding,
                max_nodes=args.max_nodes,
            )

            pyg_data = convert_to_pyg_graph(generated_graph, task_text)
            
            test_graph = TestGraph(
                domain=domain,
                llm_name=args.llm_name,
                decision_method=decision_method,
                pyg_data=pyg_data
            )
            
            answer_tasks.append(test_graph.arun(input_dict, num_rounds=1))
            metadata_for_tasks.append({"record": record, "graph": generated_graph})

        all_results = await asyncio.gather(*answer_tasks, return_exceptions=True)

        for i, result in enumerate(all_results):
            meta = metadata_for_tasks[i]
            record = meta["record"]
            
            if isinstance(result, Exception):
                print(f"Error executing task: {result}")
                is_correct = False
                processed_answer = f"ERROR: {result}"
            else:
                raw_answer = result[0] if isinstance(result, list) and result else result
                processed_answer = postprocess_answer(raw_answer)
                is_correct = check_correctness(processed_answer, record)
            
            if is_correct:
                total_correct += 1

            results_list.append({
                "record": record,
                "processed_answer": processed_answer,
                "is_correct": is_correct
            })
        
        acc = total_correct / len(results_list) * 100
        pbar.set_postfix({
            "Acc": f"{acc:.2f}% ({total_correct}/{len(results_list)})",
            "Cost": f"${Cost.instance().value:.4f}",
            "PTokens": f"{int(PromptTokens.instance().value)}"
        })

    return total_correct / len(results_list) * 100 if results_list else 0, results_list


async def main():
    args = evaluate_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading OFA model and universal roles...")
    sentence_model = SentenceTransformer(args.model_name)
    role_embeddings = {name: torch.tensor(emb) for name, emb in zip(ROLE_DESCRIPTION.keys(), sentence_model.encode(list(ROLE_DESCRIPTION.values())))}
    model = OFAModel(args, role_embeddings).to(device)
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'role_embeddings' in checkpoint:
            checkpoint.pop('role_embeddings')
        model.load_state_dict(checkpoint, strict=False)
        print(f"Model weights loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    all_dataset_logs = {}

    for dataset_name in args.dataset:
        print("\n" + "#"*60)
        print(f"### EVALUATING ON: {dataset_name.upper()} ###")
        print("#"*60 + "\n")
        
        Cost.instance().reset()
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()
        
        current_args = copy.deepcopy(args)
        current_args.dataset = dataset_name
        try:
            dataset, dataset_specific_info = get_dataset_config(dataset_name)
        except (ValueError, FileNotFoundError) as e:
            print(f"Could not load dataset {dataset_name}. Skipping. Error: {e}")
            continue

        final_accuracy, results = await evaluate(current_args, model, dataset, sentence_model, dataset_specific_info)
        total_tasks = min(len(dataset), current_args.limit) if current_args.limit is not None else len(dataset)
        log_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": "OFA",
            "dataset": dataset_name,
            "model_path": args.model_path,
            "llm_name": args.llm_name,
            "total_tasks": total_tasks,
            "accuracy": final_accuracy,
            "cost": Cost.instance().value,
            "prompt_tokens": PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value
        }
        all_dataset_logs[dataset_name] = log_record

        print("\n" + "=" * 50 + f"\nEvaluation Summary for {dataset_name.upper()}")
        print(json.dumps(log_record, indent=2))
        print("=" * 50)

        try:
            os.makedirs(os.path.dirname(args.summary_log_file), exist_ok=True)
            with open(args.summary_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_record) + '\n')
                print(f"Summary log for {dataset_name} appended to: {args.summary_log_file}")
        except Exception as e:
                print(f"Failed to write summary log file for {dataset_name}: {e}")

    print("\n\n" + "#"*60)
    print("### OVERALL EVALUATION COMPLETE ###")
    print("#"*60 + "\n")
    print("Summary of all runs:")
    for name, record in all_dataset_logs.items():
        print(f"  - {name}: Accuracy = {record['accuracy']:.2f}%, Cost = ${record['cost']:.4f}")


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
