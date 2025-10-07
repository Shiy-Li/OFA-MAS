import os
import torch
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

class OFAGraphDataset(Dataset):
    """
    Dataset to load pre-processed OFA graphs.
    Assumes all .pt files are correctly formatted with 'role_embedding' attribute.
    """
    def __init__(self, data_dir, sample_size=0):
        super(OFAGraphDataset, self).__init__()
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.graph_files = self._get_graph_files()
        
        # These are for compatibility with certain data processing functions
        self.node_label_list = [0]
        self.edge_label_list = [0]

        # Convert PyG graphs to NetworkX and store them
        self.graph_list = self._load_and_convert_graphs()

    def _get_graph_files(self):
        """
        Gets the list of graph files, optionally sampling a subset.
        """
        if not os.path.exists(self.data_dir):
            return []
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        if self.sample_size > 0:
            files = files[:self.sample_size]
        return files

    def _load_and_convert_graphs(self):
        """
        Loads PyG data objects and converts them to NetworkX graphs.
        """
        graph_list = []
        for file_name in self.graph_files:
            file_path = os.path.join(self.data_dir, file_name)
            pyg_graph = torch.load(file_path)
            
            # Convert to NetworkX, preserving node and graph attributes
            # This now strictly expects 'role_embedding' to be present.
            nx_graph = to_networkx(pyg_graph, node_attrs=['role_embedding'], to_undirected=False)

            # Manually transfer the question embedding, which is stored in 'pyg_graph.question'
            # Note: The raw question string is not expected here anymore.
            if hasattr(pyg_graph, 'question') and torch.is_tensor(pyg_graph.question):
                 # The collate_fn will look for this 'task_embedding' key
                nx_graph.graph['task_embedding'] = pyg_graph.question
            
            # Manually transfer roles to node attributes, which are also needed.
            if hasattr(pyg_graph, 'role'):
                roles = list(pyg_graph.role)
                # Append an END_TOKEN for the model to learn stopping
                roles.append('END_TOKEN')
                for i, role_name in enumerate(roles):
                    if i in nx_graph.nodes:
                        nx_graph.nodes[i]['role'] = role_name
                    elif i == len(roles) - 1: # Handle the end token which is not a real node
                        pass
            
            # Store the final roles list back into the graph object for the collate_fn
            nx_graph.graph['final_roles'] = roles

            graph_list.append(nx_graph)
        return graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx] 