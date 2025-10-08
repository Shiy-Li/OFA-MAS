import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import torch
import asyncio
from torch_geometric.data import Data

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


class Graph(ABC):

    def __init__(self,
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 optimized_spatial: bool = False,
                 fixed_spatial_masks: List[List[int]] = None,
                 node_kwargs: List[Dict] = None,
                 ):

        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i != j else 0 for j in range(len(agent_names))] for i in
                                   range(len(agent_names))]

        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        assert len(fixed_spatial_masks) == len(agent_names) * len(
            agent_names), "The fixed_spatial_masks doesn't match the number of agents"

        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.decision_node: Node = AgentRegistry.get(decision_method,
                                                     **{"domain": self.domain, "llm_name": self.llm_name})
        self.nodes: Dict[str, Node] = {}
        self.potential_spatial_edges: List[List[str, str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.init_nodes()
        self.init_potential_edges()
        
        self.spatial_logits = None
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks, requires_grad=False)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_potential_edges(self):
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id, node2_id])

    def clear_spatial_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
        
    def construct_spatial_connection_all(self, temperature: float = 1.0,
                                         threshold: float = None, ):
        self.clear_spatial_connection()

        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits,
                                                               self.spatial_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial == False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                out_node.add_successor(in_node, 'spatial')

    def to_pyg_graph(self, input: Dict[str, Any], keep_all_edge=True) -> Data:
        agent_nodes = [node for node in self.nodes.values() if node != self.decision_node]
        node_id_to_idx = {node.id: idx for idx, node in enumerate(agent_nodes)}
        node_features = []
        for node in agent_nodes:
            role = node.role
            constraint = node.constraint if hasattr(node, 'constraint') else "No constraint"
            feature = {'role': role, 'constraint': constraint}
            node_features.append(feature)
        if keep_all_edge:
            self.spatial_logits = torch.ones(len(self.potential_spatial_edges))
            self.construct_spatial_connection_all()

        edge_indices = []
        edge_weights = []
        for src_node in agent_nodes:
            src_idx = node_id_to_idx[src_node.id]
            for dst_node in src_node.spatial_successors:
                if dst_node.id in node_id_to_idx:
                    dst_idx = node_id_to_idx[dst_node.id]
                    edge_indices.append([src_idx, dst_idx])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        if edge_weights:
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        else:
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=input['task']
        )

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False


class TestGraph(ABC):
    def __init__(
            self,
            domain: str,
            llm_name: Optional[str],
            decision_method: str,
            pyg_data: Data,
            node_kwargs: List[Dict] = None,
    ):
        self.domain = domain
        self.llm_name = llm_name
        self.decision_node = AgentRegistry.get(
            decision_method, **{"domain": domain, "llm_name": llm_name}
        )
        self.nodes = {}
        self._build_from_pyg(pyg_data)

    def _build_from_pyg(self, pyg_data: Data):
        roles = [d['role'] for d in pyg_data.x]
        constraints = [d.get('constraint') for d in pyg_data.x]
        if self.domain == 'mmlu':
            agent_type = 'AnalyzeAgent'
        elif self.domain == 'humaneval':
            agent_type = 'CodeWriting'
        elif self.domain == 'gsm8k':
            agent_type = 'MathSolver'
        elif self.domain == 'aqua':
            agent_type = 'MathSolver'
        else:
            agent_type= 'AnalyzeAgent'
        prompt_set = PromptSetRegistry.get(self.domain)
        for idx, (role, constraint) in enumerate(zip(roles, constraints)):
            actual_constraint = constraint
            agent_kwargs = {
                'role': role,
                'domain': self.domain,
                'llm_name': self.llm_name,
                'constraint': actual_constraint
            }
            if self.domain == 'mmlu':
                agent_kwargs['constraint'] = actual_constraint
            agent = AgentRegistry.get(
                agent_type,
                **agent_kwargs
            )
            self.add_node(agent)

        edge_index = pyg_data.edge_index.cpu().numpy().T
        node_list = list(self.nodes.values())

        for src_idx, dst_idx in edge_index:
            src_node = node_list[src_idx]
            dst_node = node_list[dst_idx]
            src_node.add_successor(dst_node, "spatial")

    def add_node(self, node: Node):
        node_id = node.id if node.id else shortuuid.ShortUUID().random(length=4)
        self.nodes[node_id] = node
        return node

    async def arun(self, inputs: Dict[str, Any], num_rounds=1, max_tries: int = 3, max_time: int = 600) -> List[Any]:

        for round in range(num_rounds):

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if
                                    deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(inputs),
                                               timeout=max_time)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers

    def connect_decision_node(self):
        for node in self.nodes.values():
            node.add_successor(self.decision_node)

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()

    def construct_temporal_connection(self, round: int = 0, temperature: float = 1.0,
                                      threshold: float = None, ):
        self.clear_temporal_connection()

        if round == 0:
            return 0

    def clear_temporal_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []


def min_max_norm(tensor: torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
