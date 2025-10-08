import math
import sys
import os
import torch
import pickle
import random
import json
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

def get_kwargs(mode: str, N: int):
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks = None
    fixed_temporal_masks = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj = [[0] * N for _ in range(N)]
        base = N // layer_num
        rem = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base + (1 if i < rem else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj[i][j] = 1
        return adj

    def generate_mesh_graph(N):
        if N > 4 and int(math.sqrt(N))**2 == N:
            size = int(math.sqrt(N))
            adj = [[0] * N for _ in range(N)]
            for i in range(N):
                if (i + 1) % size != 0:
                    adj[i][i+1] = adj[i+1][i] = 1
                if i < N - size:
                    adj[i][i+size] = adj[i+size][i] = 1
            return adj
        return [[1 if i != j else 0 for i in range(N)] for j in range(N)]

    def generate_star_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj[0][i] = adj[i][0] = 1
        return adj

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Normal'}]
    elif mode in ('FullConnected', 'FakeFullConnected', 'FakeAGFull'):
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode in ('Random', 'FakeRandom', 'FakeAGRandom'):
        fixed_spatial_masks = [[random.randint(0,1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0,1) for _ in range(N)] for _ in range(N)]
    elif mode in ('Chain', 'FakeChain'):
        fixed_spatial_masks = [[1 if abs(i-j)==1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==j else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
    elif mode in ('Mesh', 'FakeMesh'):
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
    elif mode in ('Star', 'FakeStar'):
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]

    elif 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }


def save_graph_with_features(flow_graph, filepath, metadata):
    for key, value in metadata.items():
        setattr(flow_graph, key, value)
    torch.save(flow_graph, filepath)


class Accuracy:
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0

    def update(self, predicted: str, target: str) -> bool:
        is_correct = predicted == target
        self._num_correct += int(is_correct)
        self._num_total += 1
        return is_correct

    def get(self) -> float:
        return self._num_correct / self._num_total if self._num_total > 0 else 0.0

    def print(self):
        acc = self.get() * 100
        print(f"Accuracy: {acc:.1f}% ({self._num_correct}/{self._num_total})")
