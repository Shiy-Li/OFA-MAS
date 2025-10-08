import math
from typing import Iterator, Any, List
from tqdm import tqdm
import copy
import sys
import os
import torch
import asyncio
import argparse
import random
import pickle
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from experiment.OFA.utils import get_kwargs, Accuracy
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from GDesigner.graph.graph import Graph, TestGraph
from experiment.OFA.uni_role import ROLE_DESCRIPTION


def parse_args():
    parser = argparse.ArgumentParser(description="Generate supervised fine-tuning data for OFA on MMLU.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for evaluation")
    parser.add_argument('--num_iterations', type=int, default=5, help="Number of iterations to define training set size (batch_size * num_iterations)")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'], help='Base agent name')
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds for each query")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu", help="Domain name, same as dataset name")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method for the final node")
    return parser.parse_args()


OUTPUT_DIR = "./Finetune_OFA_mmlu"


def get_all_classic_configs():
    configs = set()
    for agent_num in range(2, 7):
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


async def main():
    args = parse_args()
    download()

    print("Initializing sentence model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    EMBEDDINGS_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.', 'precomputed_role_embeddings.pkl')
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

    dataset = MMLUDataset('dev')
    train_set_size = args.batch_size * args.num_iterations
    train_set_size = min(train_set_size, len(dataset))

    train_indices = list(range(train_set_size))
    finetune_dataset = torch.utils.data.Subset(dataset, train_indices)

    print(f"Loaded {len(finetune_dataset)} tasks for fine-tuning data generation.")

    configs = get_all_classic_configs()
    print(f"Generating fine-tuning data for {len(configs)} classic configurations...")

    for mode, agent_num in configs:
        print(f"\n=== Processing configuration: Mode={mode}, Agent Nums={agent_num} ===")

        all_kwargs = get_kwargs(mode, agent_num)
        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)

        graph_kwargs = {}
        if 'fixed_spatial_masks' in all_kwargs:
            graph_kwargs['fixed_spatial_masks'] = all_kwargs['fixed_spatial_masks']

        graph_kwargs['node_kwargs'] = [{'role': role, 'constraint': ROLE_DESCRIPTION[role]} for role in random_roles]

        graph = Graph(
            domain=args.domain,
            llm_name=args.llm_name,
            agent_names=[args.agent_names[0]] * agent_num,
            decision_method=args.decision_method,
            **graph_kwargs
        )

        await evaluate_and_save(
            graph=graph,
            dataset=finetune_dataset,
            num_rounds=args.num_rounds,
            eval_batch_size=args.batch_size,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num,
            role_embeddings_dict=role_embeddings_dict,
            sentence_model=sentence_model
        )

    print("All fine-tuning data generation for MMLU is complete.")


async def evaluate_and_save(
        graph: Graph,
        dataset,
        num_rounds: int,
        eval_batch_size: int,
        args,
        current_mode: str,
        current_agent_num: int,
        role_embeddings_dict: dict,
        sentence_model
):
    accuracy = Accuracy()
    original_dataset = dataset.dataset

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for record in dataset:
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    num_batches = math.ceil(len(dataset) / eval_batch_size)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pbar = tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches, desc=f"Evaluating {current_mode}-{current_agent_num}")
    for i_batch, record_batch in pbar:
        tasks = []
        metadata_list = []

        for record in record_batch:
            g_copy = copy.deepcopy(graph)
            input_dict = original_dataset.record_to_input(record)

            pyg_data = g_copy.to_pyg_graph(input_dict)

            roles = [d['role'] for d in pyg_data.x]
            embeddings = [role_embeddings_dict[r] for r in roles]
            pyg_data.role = roles
            pyg_data.role_embedding = torch.stack(embeddings)

            question_text = input_dict['task']
            pyg_data.question = sentence_model.encode(question_text, convert_to_tensor=True)

            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=pyg_data
            )
            tasks.append(asyncio.create_task(tg.arun(input_dict, num_rounds)))
            metadata_list.append({
                "record": record,
                "pyg_data": pyg_data,
                "question": input_dict['task']
            })

        raw_results = await asyncio.gather(*tasks)

        for i, raw_answer in enumerate(raw_results):
            meta = metadata_list[i]
            record = meta['record']

            answer = original_dataset.postprocess_answer(raw_answer)
            correct_answer = original_dataset.record_to_target_answer(record)
            is_correct = accuracy.update(answer, correct_answer)

            if is_correct:
                record_id = record.get('id', f"task_{i_batch * eval_batch_size + i}")
                name = "_".join(map(str, ['mmlu', record_id, current_mode, current_agent_num, is_correct]))
                filepath = os.path.join(OUTPUT_DIR, f'{name}.pt')

                attributes_to_save = {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_correct,
                }
                for key, value in attributes_to_save.items():
                    setattr(meta['pyg_data'], key, value)
                torch.save(meta['pyg_data'], filepath)

        pbar.set_postfix({"Accuracy": f"{accuracy.get():.2f}%"})

    print(f"Finished Mode={current_mode}, Agent Nums={current_agent_num}, Accuracy: {accuracy.get():.2f}%")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())