import os
import json
import math
import asyncio
import copy
import sys
import argparse
import random
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

sys.stdout.reconfigure(encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GDesigner.graph.graph import Graph, TestGraph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.tools.coding.python_executor import PyExecutor
from experiment.OFA.utils import get_kwargs
from experiment.OFA.uni_role import ROLE_DESCRIPTION

OUTPUT_DIR = "./Finetune_OFA_humaneval"
TASK_SPLIT_FILE = "./task_split_humaneval.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate supervised fine-tuning data for OFA on HumanEval.")
    parser.add_argument('--dataset_json', type=str, default="../../../datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'], help='Base agent name')
    parser.add_argument('--decision_method', type=str, default="FinalWriteCode", help="Decision method")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--num_iterations', type=int, default=5, help="Number of iterations to define training set size (batch_size * num_iterations)")
    parser.add_argument('--domain', type=str, default="humaneval", help="Task domain")
    return parser.parse_args()


def get_all_classic_configs():
    configs = set()
    for agent_num in range(2, 6):
        if agent_num == 2:
            configs.add(('Chain', 2))
        elif agent_num == 3:
            configs.add(('Chain', 3))
            configs.add(('Star', 3))
            configs.add(('FullConnected', 3))
        else:
            configs.add(('Chain', agent_num))
            configs.add(('Star', agent_num))
            configs.add(('Layered', agent_num))
            configs.add(('FullConnected', agent_num))
            configs.add(('Mesh', agent_num))
    return list(configs)


async def evaluate_and_save(
        graph: Graph,
        dataset,
        args,
        current_mode: str,
        current_agent_num: int,
        role_embeddings_dict: dict,
        sentence_model
):
    executor = PyExecutor()
    num_batches = math.ceil(len(dataset) / args.batch_size)
    total_solved = 0

    pbar = tqdm(range(num_batches), desc=f"Processing {current_mode}-{current_agent_num}")
    for i_batch in pbar:
        batch_records = dataset[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not batch_records:
            continue

        tasks = []
        for record in batch_records:
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["prompt"]}
            pyg_data = realized_graph.to_pyg_graph(input_dict)
            roles = [d['role'] for d in pyg_data.x]
            embeddings = [role_embeddings_dict[r] for r in roles]
            pyg_data.role = roles
            pyg_data.role_embedding = torch.stack(embeddings)
            question_text = input_dict['task']
            pyg_data.question = sentence_model.encode(question_text, convert_to_tensor=True)

            tg = TestGraph(domain=args.domain, llm_name=args.llm_name, 
                           decision_method=args.decision_method, pyg_data=pyg_data)
            metadata = {
                "record": record,
                "pyg_data": pyg_data,
                "question": record["prompt"],
            }
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        coroutines_to_run = [task for task, _ in tasks]
        results = await asyncio.gather(*coroutines_to_run, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task execution error: {result}")
                continue

            metadata = tasks[i][1]
            record = metadata['record']

            raw_answer = result[0] if isinstance(result, list) and result else result
            answer_code = raw_answer.lstrip("```python\n").rstrip("\n```")

            is_solved, _, _ = executor.execute(answer_code, [record["test"]], timeout=10)

            if is_solved:
                total_solved += 1
                record_id = record.get('task_id', f"task_{i_batch * args.batch_size + i}")
                name = "_".join(map(str, ['humaneval', record_id, current_mode, current_agent_num, is_solved]))
                filepath = os.path.join(OUTPUT_DIR, f'{name}.pt')

                attributes_to_save = {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_solved,
                }
                for key, value in attributes_to_save.items():
                    setattr(metadata['pyg_data'], key, value)
                torch.save(metadata['pyg_data'], filepath)
        
        pbar.set_postfix({"Solved": f"{total_solved}"})

    print(f"Config {current_mode}-{current_agent_num} finished. Solved {total_solved} / {len(dataset)} tasks.")


async def main():
    args = parse_args()
    
    print("Initializing sentence model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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

    dataset = list(JSONLReader.parse_file(args.dataset_json))

    train_set_size = args.batch_size * args.num_iterations
    train_set_size = min(train_set_size, len(dataset))
    
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:train_set_size]
    test_indices = all_indices[train_set_size:]

    with open(TASK_SPLIT_FILE, 'w') as f:
        json.dump({
            "train_indices": train_indices,
            "test_indices": test_indices
        }, f)
    print(f"Saved new task split to {TASK_SPLIT_FILE} ({len(train_indices)} train, {len(test_indices)} test)")

    finetune_dataset = [dataset[i] for i in train_indices]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loaded {len(finetune_dataset)} tasks for fine-tuning data generation.")
    
    configs = get_all_classic_configs()
    print(f"Generating HumanEval fine-tuning data for {len(configs)} configurations...")

    for mode, agent_num in configs:
        print(f"\n=== Configuration: Mode={mode}, Agent Nums={agent_num} ===")
        all_kwargs = get_kwargs(mode, agent_num)
        
        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)

        graph_kwargs = {}
        if 'fixed_spatial_masks' in all_kwargs:
            graph_kwargs['fixed_spatial_masks'] = all_kwargs['fixed_spatial_masks']
        
        graph_kwargs['node_kwargs'] = [{'role': role, 'constraint': ROLE_DESCRIPTION[role]} for role in random_roles]
        
        graph = Graph(domain=args.domain,
                      llm_name=args.llm_name,
                      agent_names=[args.agent_names[0]] * agent_num,
                      decision_method=args.decision_method,
                      **graph_kwargs)

        await evaluate_and_save(
            graph=graph,
            dataset=finetune_dataset,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num,
            role_embeddings_dict=role_embeddings_dict,
            sentence_model=sentence_model
        )
        
    print("\nAll HumanEval fine-tuning data generation completed!")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())