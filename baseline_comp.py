import numpy as np
from load_data import load_data
import torch
from modules import GNN
from train_model import train_model
from subgraph_relevance import subgraph_original, subgraph_mp_transcription, subgraph_mp_forward_hook, get_H_transform
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pandas as pd
from io import StringIO
import pickle as pkl
import torch.nn.functional as F

def sigm(z):
    return torch.tanh(0.5*z)*0.5+0.5

def gnnexplainer(g,nn,H0=None,steps=500,lr=0.5,lambd=0.01,verbose=False):
    z = torch.ones(g.get_adj().shape)*g.get_adj()*2
    num_layer = len(nn.blocks) -1
    bar = tqdm(range(steps)) if verbose else range(steps)
    for i in bar:
        z.requires_grad_(True)

        score = nn.forward(g.get_adj(),H0=H0,masks=[sigm(z)]*num_layer)[g.label] # ,sigm(z)

        emp   = -score
        reg   = lambd*((z)**2).sum() # torch.zeros((1,))   

        if i in [j**3 for j in range(100)] and verbose: print('%5d %8.3f %8.3f'%(i,emp.item(),reg.item()))
        
        (emp+reg).backward()

        with torch.no_grad():
            z = (z - lr*z.grad)
        z.grad = None

    return z.data

from captum.attr import IntegratedGradients
def get_top_edges_edge_ig(nn, g, target, edge_num, drop_selfloop=False):
    def model_edge_forward(edge_mask, nn, g):
        edges = g.get_adj().nonzero()
        a = torch.zeros_like(g.get_adj())
        for mask, edge in zip(edge_mask, edges):
            if edge[0] > edge[1]: continue
            a[edge[0]][edge[1]] = mask
            a[edge[1]][edge[0]] = mask
        pred = nn.forward(a, H0=g.node_features)
        return pred.reshape(1,2)

    ig = IntegratedGradients(model_edge_forward)
    if drop_selfloop:
        edges = (g.get_adj()-torch.eye(g.nbnodes)).nonzero()
        input_mask = torch.ones(len((g.get_adj()-torch.eye(g.nbnodes)).nonzero())).requires_grad_(True)
    else:
        edges = g.get_adj().nonzero()
        input_mask = torch.ones(len(g.get_adj().nonzero())).requires_grad_(True)
    ig_mask = ig.attribute(input_mask, target=target, additional_forward_args=(nn, g),
                            internal_batch_size=len(input_mask))

    edge_mask = ig_mask.cpu().detach().numpy()

    edges_sort = []
    for i in (-edge_mask).argsort()[:edge_num]:
        edges_sort.append(tuple(edges[i].tolist()))
    return edges_sort
