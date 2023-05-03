from typing import Dict, List
import torch
import numpy as np
from functools import reduce
import time
from modules import GNN

def subgraph_original(nn: GNN, g, S: List, alpha: float = 0.0, gamma: List = None, verbose: bool = False):
    ####################
    ###### GNN-LRP-naive
    ####################

    g.get_adj() # in case g's adj doesn't exist

    # Overhead: Get walk relevance matrix
    if verbose: time_begein = time.time()

    nbnodes = len(g.get_adj())
    num_layer = len(nn.blocks)
    AA = nn.adj(g.get_adj())
    R = torch.zeros([len(S)]*(num_layer))
    if not gamma: gamma = np.linspace(3,0,num_layer-1)

    ## Make the walk-grid:
    if alpha == 0: grid = [S]*(num_layer -1)
    else: grid = [np.arange(nbnodes)]*(num_layer -1)
    grid = np.meshgrid(*grid)
    grid = [I.flatten() for I in grid]
    rel = 0
    mask = torch.full([nbnodes], alpha)
    mask[S] = 1
    if alpha != 0:
        mask_alpha = torch.zeros([nbnodes])
        mask_alpha[S] = 1.
    for w in zip(*grid):
        R_w = nn.lrp(g.get_adj(),gamma,g.label,tuple(w), g.node_features)

        if alpha == 0:
            for i, j in enumerate(S):
                R[(i,)+w] = R_w[j]
            # rel += mask @ R_w
        else:
            pow_nb = 0
            for node in w:
                if not node in S: pow_nb += 1
            if pow_nb == len(w):
                rel += (alpha ** pow_nb) * (mask_alpha @ R_w)
            else:
                rel += (alpha ** pow_nb) * (mask @ R_w)

    if verbose: 
        time_overhead = time.time() - time_begein
        time_begein = time.time()

    if alpha == 0:
        rel = R.sum()
        if verbose: 
            time_subrel = time.time() - time_begein
            print(f"original\tnbnodes: {nbnodes},\tlayers: {num_layer-1},\toverhead: {time_overhead:.6f},\tsubrel: {time_subrel:.6f}")
    return rel


def softmin(b): return -0.5*torch.log(1.0+torch.exp(-2*b))

def get_H_transform(A, nn, H0=None, gammas=None, mode='gamma'):
    model_depth = len(nn.blocks)-1
    if gammas is None:
        gammas = np.linspace(3,0,model_depth)

    masks = [1]*(len(nn.blocks)-1)

    H0 = nn.ini(A, H0)
    H = nn.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0)).detach()
    A = nn.adj(A)

    transforms = []

    block_no = 0
    for l,mask in zip(nn.blocks[1:],masks):
        block_no += 1
        gamma = gammas[block_no - 1]

        Hin = H
        n = Hin.shape[0]

        transform_block = None

        for Wo,Bo,Ao in zip(l.W,l.B,[A*mask]+[torch.eye(n).reshape(1,n,n)]*(len(l.W)-1)):

            Hout = 0
            weight = 0
            bias = 0

            for ao,wo,bo in zip(Ao,Wo.data,Bo.data):
                bo = softmin(bo)
                Hout = Hout + ao.permute(1,0).matmul(Hin).matmul(wo) + bo
                weight += (ao.unsqueeze(-1).unsqueeze(-1) * wo.unsqueeze(0).unsqueeze(0)).permute(0,2,1,3)
                bias += bo

            if mode == "gamma":
                bias = bias + gamma * bias.clamp(min=0)
                transform = (weight + gamma * weight.clamp(min=0)) * Hin.unsqueeze(-1).unsqueeze(-1)
            elif mode == "abs":
                bias = bias.abs()
                transform = (weight * Hin.unsqueeze(-1).unsqueeze(-1)).abs()
            elif mode == "clip-pos":
                bias = torch.clip(bias.abs(), min=0)
                transform = torch.clip(weight * Hin.unsqueeze(-1).unsqueeze(-1), min=0)
            elif mode == "none":
                bias = bias
                transform = weight * Hin.unsqueeze(-1).unsqueeze(-1)
                
            transform = (transform / ((transform.sum(axis=0).sum(axis=0) + bias).unsqueeze(0).unsqueeze(0))).reshape([-1,n*transform.shape[-1]])

            if transform_block == None:
                transform_block = transform
            else:
                transform_block = transform_block @ transform

            if Wo.shape[2] > 10:
                Hin = Hout.clamp(min=0)
            else:
                Hin = Hout

        transforms.append(transform_block.reshape([n,transform_block.shape[0]//n,n,transform_block.shape[1]//n]))

        H = Hin

    return H.data / 20**.5, transforms


def subgraph_mp_transcription(nn: GNN, g, S: List, alpha: float = 0.0, gamma: List = None, verbose: bool = False, H=None, transforms=None, target=None):

    if verbose:
        time_begin = time.time()

    if H is None and transforms is None:
        H, transforms = get_H_transform(g.get_adj(), nn, g.node_features, gamma)
        
    if verbose:
        time_overhead = time.time() - time_begin
        time_begin = time.time()

    if target is None:
        target = nn.forward(g.get_adj(), g.node_features).argmax()

    nbnodes = H.shape[0]
    mask = torch.full([nbnodes], alpha)
    mask[list(S)] = 1.0
    mask = torch.diag(mask)

    relevance_subgraph = H @ torch.diag(torch.eye(H.shape[1])[target])
    for transform in reversed(transforms):
        # einsum slow
        # relevance_subgraph = torch.einsum('ijkl,kl->ij', transform, mask @ relevance_subgraph)

        nbneurons_in = transform.shape[1]
        nbneurons_out = transform.shape[3]
        transform = transform.reshape(nbnodes * nbneurons_in, nbnodes * nbneurons_out)

        relevance = (mask @ relevance_subgraph).reshape([nbnodes * nbneurons_out, 1])
        relevance = (transform @ relevance).reshape(nbnodes, nbneurons_in)
        relevance_subgraph = relevance
        
    
    rel_1 = (mask @ relevance_subgraph).sum()
    
    if alpha == 0.0: 
        if verbose:
            time_subrel = time.time() - time_begin
            print(f"mp_transc\tnbnodes: {nbnodes},\tlayers: {len(nn.blocks)-1},\toverhead: {time_overhead:.6f},\tsubrel: {time_subrel:.6f}")
        return rel_1

    mask = torch.full([nbnodes], alpha)
    mask[list(S)] = 0.0
    mask = torch.diag(mask)
    relevance_subgraph = H @ torch.diag(torch.eye(H.shape[1])[target])
    for transform in reversed(transforms):
        # relevance_subgraph = torch.einsum('ijkl,kl->ij', transform, mask @ relevance_subgraph)

        nbneurons_in = transform.shape[1]
        nbneurons_out = transform.shape[3]
        transform = transform.reshape(nbnodes * nbneurons_in, nbnodes * nbneurons_out)

        relevance = (mask @ relevance_subgraph).reshape([nbnodes * nbneurons_out, 1])
        relevance = (transform @ relevance).reshape(nbnodes, nbneurons_in)
        relevance_subgraph = relevance
        
    rel_2 = (mask @ relevance_subgraph).sum()

    if verbose:
        time_subrel = time.time() - time_begin
        print(f"mp_transc\tnbnodes: {nbnodes},\tlayers: {len(nn.blocks)-1},\toverhead: {time_overhead:.6f},\tsubrel: {time_subrel:.6f}")

    return rel_1 - rel_2


def subgraph_mp_forward_hook(nn: GNN, g: Dict, S: List, alpha: float = 0.0, gamma: List = None, verbose: bool = False):
    A = g.get_adj()
    t = nn.forward(g.get_adj(), g.node_features).argmax()
    model_depth = len(nn.blocks)-1
    gammas = np.linspace(3,0,model_depth)

    
    H0 = nn.ini(A, g.node_features)
    A = nn.adj(A)

    if verbose: 
        time_begin = time.time()
    H1 = nn.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0)).data
    H1.requires_grad_(True)
    H = H1

    # M = np.zeros(H.shape[0])
    M = np.full(H.shape[0], alpha)
    
    M[list(S)] = 1
    M = torch.FloatTensor(M.reshape(-1,1))

    for l,gamma in zip(nn.blocks[1:],gammas):
        H = l.lrpforward(H,A,gamma)
        H = H * M + (1-M) * (H.data)

    H = H.sum(dim=0) / 20**.5
    
    if verbose:
        time_forward = time.time() - time_begin
        time_begin = time.time()

    H[t].backward()
    rel_1 = (H1*H1.grad).sum(dim=1).data @ M

    if verbose:
        time_backward = time.time() - time_begin

    
    if alpha == 0.0:
        if verbose: print(f"forward_hook\tnbnodes: {len(g.get_adj())},\tlayers: {len(nn.blocks)-1},\tforward: {time_forward:.6f},\tbackward: {time_backward:.6f}")
        return rel_1[0]
    
    if verbose: time_begin = time.time()
    # H1.zero_grad()
    H1 = nn.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0)).data
    H1.requires_grad_(True)
    H = H1

    # M = np.zeros(H.shape[0])
    M = np.full(H.shape[0], alpha)
    
    M[list(S)] = 0
    M = torch.FloatTensor(M.reshape(-1,1))

    for l,gamma in zip(nn.blocks[1:],gammas):
        H = l.lrpforward(H,A,gamma)
        H = H * M + (1-M) * (H.data)

    H = H.sum(dim=0) / 20**.5
    
    if verbose:
        time_forward2 = time.time() - time_begin
        time_begin = time.time()

    H[t].backward()
    rel_2 = (H1*H1.grad).sum(dim=1).data @ M

    if verbose:
        time_backward2 = time.time() - time_begin
        print(f"forward_hook\tnbnodes: {g.get_adj().shape[0]},\tlayers: {len(nn.blocks)-1},\tforward1: {time_forward:.6f},\tbackward1: {time_backward:.6f},\tforward2: {time_forward2:.6f},\tbackward2: {time_backward2:.6f}")

    return (rel_1 - rel_2)[0]
