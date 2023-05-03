import numpy as np
from load_data import load_data
import torch
from modules import GNN
from train_model import train_model
from subgraph_relevance import subgraph_original, subgraph_mp_transcription, subgraph_mp_forward_hook, get_H_transform
from utils import *
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pandas as pd
from io import StringIO
import pickle as pkl
import json

def walk_rel(transforms, rel, walk, mode="neuron"):
    """Get the relevance score of the walk

    Args:
        transforms (List[torch.Tensor]): The list of transition matrices
        rel (torch.Tensor): The initial relevance, shaped (n, d_{L})
        walk (Tuple(int)): The walk
        mode (str, optional): Is the walk neuron-level or node-level. Defaults to "neuron".

    Returns:
        float: The relevance score of the walk
    """    
    if mode == "neuron" or mode == "node-max-neuron":
        rel = rel.copy().flatten()[walk[-1]]
        for step in reversed(range(len(transforms))):
            rel = transforms[step].reshape(transforms[step].shape[0]*transforms[step].shape[1], -1)[walk[step], walk[step+1]] * rel
        return float(rel)
    elif mode == "node":
        rel = rel.copy()[walk[-1]]
        for step in reversed(range(len(transforms))):
            rel = transforms[step][walk[step], :, walk[step+1], :] @ rel
        rel = rel.sum()
        return float(rel)

def transform_node_neuron_to_global_neuron(walk, transforms):
    """Transform the [node, neuron] representations walks into neuron-level.

    Args:
        walk (Tuple(int)): The [node, neuron] representations to be transformed.
        transforms (List[torch.Tensor]): The list of transition matrices

    Returns:
        Tuple(Tuple(int)): _description_
    """    
    # Transform back to node & neurons
    global_node_neuron = []
    for n, t in zip(walk, transforms):
        global_node_neuron.append(t.shape[1] * n[0] + n[1])
        
    global_node_neuron.append(transforms[-1].shape[3] * walk[-1][0] + walk[-1][1])
                            
    return tuple(global_node_neuron) # [[node, neuron], [node, neuron], [node, neuron], ...]

def transform_global_neuron_to_node_neuron(walk, transforms):
    """Transform the neuron-level walks into [node, neuron] representations.

    Args:
        walk (Tuple(int)): The neuron-level walk to be transformed.
        transforms (List[torch.Tensor]): The list of transition matrices

    Returns:
        Tuple(Tuple(int)): _description_
    """    
    # Transform back to node & neurons
    walk_node_neuron = []
    for n, t in zip(walk, transforms):
        walk_node_neuron.append((int(torch.div(n, t.shape[1], rounding_mode='floor')), int(n % t.shape[1])))
        
    walk_node_neuron.append((int(torch.div(walk[-1], transforms[-1].shape[3], rounding_mode='floor')), 
                                int(walk[-1] % transforms[-1].shape[3])))
                            
    return tuple(walk_node_neuron) # [[node, neuron], [node, neuron], [node, neuron], ...]

def subwalk_exist(subwalk, max_walks):
    res = False
    for max_walk in max_walks:
        i = 0
        while i < len(subwalk):
            if max_walk[i] != subwalk[i]:
                break
            i += 1
        if i == len(subwalk): res = True
    return res

def topk_walks(g, nn, num_walks=300, lrp_mode="clip-pos", mode="neuron", negative_transition_strategy="none", H=None, transforms=None, verbose=False, how_max='none', split_pos_neg=True):
    """Get top-k most relevant walks using max-sum algorithm.

    Args:
        g (Graph): Input graph
        nn (torch.Module): The GNN model to be explained
        num_walks (int, optional): Number of top walks to output. Defaults to 300.
        lrp_mode (str, optional): Mode of LRP. Defaults to "abs".
        mode (str, optional): "neuron" for neuron-level walks, "node" for approx. node-level walks, "node-top-neuron" for using the relevance of the top neuron-level walk as the node-level walk's relevance score. Defaults to "neuron".
        H (numpy.array, optional): Last layer's activation, shaped (n, d). Defaults to None.
        transforms (List[numpy.array], optional): The list of transition matrices. Defaults to None.
        verbose (bool, optional): True if output additional information and output time consumption . Defaults to False.

    Returns:
        List(Tuple(int)): List of top-k walks.
        If verbose == True: List of top-k walks & List of time consumption w.r.t. number of walks.
    """    

    if verbose: 
        print("Make sure that you receive two outputs: List of top-k walks, List of time consumption")
        time_list = []

    if H is None or transforms is None:
        H, transforms = get_H_transform(g.get_adj(),nn, H0=g.node_features, gammas=None, mode=lrp_mode)
        H = H.numpy()
    else:
        if isinstance(H, torch.Tensor): H = H.numpy()
    
    num_walks = min(num_walks, len(g.get_adj()) ** (len(transforms) + 1) if g.node_features is None else (g.node_features.shape[0] * g.node_features.shape[1]) ** (len(transforms) + 1))

    # Modify transforms in case of negative values
    if negative_transition_strategy == "abs":
        transforms_modified = [np.abs(transform.numpy()) for transform in transforms]
    elif negative_transition_strategy == "clip-pos":
        transforms_modified = [np.clip(transform.numpy(), a_min=0, a_max=None) for transform in transforms]
    elif negative_transition_strategy == "none":
        transforms_modified = [transform.numpy() for transform in transforms]
    elif negative_transition_strategy == "min_offset":
        transforms_modified = [transform.numpy() for transform in transforms]
    else:
        raise NotImplementedError()
    ########### Find the top-1 walk
    if verbose: time_a = time.time()

    max_mapping = []

    # Get prediction class
    pred = nn.forward(g.get_adj(), H0=g.node_features).argmax()
    init_rel = np.zeros_like(H)
    init_rel[:, pred] = H[:, pred]
    
    if negative_transition_strategy == "abs":
        init_rel = np.abs(init_rel) # should be absolute value
    elif negative_transition_strategy == "clip-pos":
        init_rel = np.clip(init_rel, a_min=0, a_max=None) # should be clipped to non-negative
    elif negative_transition_strategy == "none":
        init_rel = init_rel # do nothing
    elif negative_transition_strategy == "min_offset":
        real_init_rel = init_rel
        init_rel = init_rel - np.min(init_rel, axis=0)[np.newaxis,:]
    
    max_rel_step = init_rel

    # Compute the max walk mapping by max-sum
    if mode == "neuron" or mode == "node-max-neuron":
        for i in range(1,1+len(transforms_modified)):
            rel_step = transforms_modified[-i] * max_rel_step
            max_mapping = [rel_step.reshape(rel_step.shape[0] * rel_step.shape[1], -1).argmax(axis=1)] + max_mapping
            max_rel_step = rel_step.reshape(rel_step.shape[0] * rel_step.shape[1], -1)\
                                .max(axis=1).reshape(rel_step.shape[0], rel_step.shape[1])
    else:
        for i in range(1,1+len(transforms_modified)):
            # rel_step = transforms_modified[-i] * max_rel_step
            # rel_step = np.einsum("ijkl,kl->ik", transforms_modified[-i], max_rel_step)
            # assert np.isclose(rel_step.sum(axis=-1).sum(axis=-2), rel_step_).all()
            
            # mapping = rel_step.sum(axis=-1).sum(axis=-2).argmax(axis=1)
            # mapping = rel_step.argmax(axis=1)
            if how_max == 'abs':
                rel_step = np.einsum("ijkl,kl->ik", transforms_modified[-i], max_rel_step)
                mapping = abs(rel_step).argmax(axis=1)
            elif how_max == 'l1norm':
                rel_step = np.einsum("ijkl,kl->ijk", transforms_modified[-i], max_rel_step)
                rel_step = abs(rel_step).sum(axis=1)
                mapping = rel_step.argmax(axis=1)
            elif how_max == 'l2norm':
                rel_step = np.einsum("ijkl,kl->ijk", transforms_modified[-i], max_rel_step)
                rel_step1 = abs(rel_step).sum(axis=1)
                mapping1 = rel_step1.argmax(axis=1)
                
                rel_step = np.power(rel_step, 2).sum(axis=1)
                mapping = rel_step.argmax(axis=1)
                # print((mapping1 == mapping).mean())

            else:
                rel_step = np.einsum("ijkl,kl->ik", transforms_modified[-i], max_rel_step)
                mapping = rel_step.argmax(axis=1)

            max_mapping = [mapping] + max_mapping

            max_rel_step_ = np.zeros([g.nbnodes, transforms_modified[-i].shape[1]])
            
            for j in range(g.nbnodes):
                max_rel_step_[j] = transforms_modified[-i][j,:,mapping[j],:] @ max_rel_step[mapping[j]]
                # rel_step[j,:,0:mapping[j],:] = 0
                # if mapping[j] < rel_step.shape[2]:
                #     rel_step[j,:,mapping[j]+1:,:] = 0
            # max_rel_step = rel_step.sum(axis=-1).sum(axis=-1)
            max_rel_step = max_rel_step_
            
        max_rel_step = max_rel_step.sum(axis=-1)

    # Get the top-1 walk
    max_walk = [max_rel_step.argmax()]
    for mapping in max_mapping:
        max_walk += [mapping[max_walk[-1]]] 

    max_walk = tuple(max_walk)
    if verbose: time_list.append(time.time() - time_a)

    ########### Find the following top walks

    # Get number of neurons for each layer
    if mode == "neuron" or mode == "node-max-neuron":
        nb_neurons_list = [transforms_modified[0].shape[0] * transforms_modified[0].shape[1]]
        nb_neurons_list = nb_neurons_list + [transforms_modified[i].shape[2] * transforms_modified[i].shape[3] for i in range(len(transforms_modified))]
    else:
        nb_nodes = H.shape[0]

    top_k_max_walks = [tuple(max_walk)] # to store the top walks
        
    last_max_walk = max_walk
    sub_walk_rel_dict = {}
    for i in range(len(max_walk)+1):
        sub_walk_rel_dict[tuple(max_walk[:i+1])] = 0

    if mode == 'node-max-neuron':
        top_k_max_walks_node = [tuple([node_neuron[0] for node_neuron in transform_global_neuron_to_node_neuron(max_walk, transforms_modified)])]

    while True:
    # for _ in range(num_walks-1):
        if mode == 'node-max-neuron':
            if len(top_k_max_walks_node) == num_walks or len(top_k_max_walks) > 5*num_walks: break
        else:
            if len(top_k_max_walks) == num_walks: break

        if verbose: time_a = time.time()
        
        for idx in range(len(last_max_walk)):
            walk = np.array(last_max_walk[:idx], dtype=int)

            if mode == "neuron" or mode == "node-max-neuron":
                nb_features = nb_neurons_list[idx]
            else:
                nb_features = nb_nodes

            for feature in range(nb_features):
                sub_walk = np.append(walk, feature)
                if tuple(sub_walk) in sub_walk_rel_dict.keys() or tuple(sub_walk) in top_k_max_walks: continue # prevent duplicate computation

                # check if this step exists
                if len(walk) > 0:
                    if mode == "neuron" or mode == "node-max-neuron":
                        if len(walk) == 1:
                            last_node = torch.div(walk[-1], transforms_modified[0].shape[1], rounding_mode='floor')
                        else:
                            last_node = torch.div(walk[-1], transforms_modified[len(walk)-2].shape[3], rounding_mode='floor')

                        this_node = torch.div(feature, transforms_modified[len(walk)-1].shape[3], rounding_mode='floor')
                    else:
                        last_node, this_node = walk[-1], feature
                        
                    if g.get_adj()[last_node, this_node] == 0: continue

                tmp_walk = sub_walk
                for mapping in max_mapping[idx:]:
                    tmp_walk = np.append(tmp_walk, mapping[tmp_walk[-1]]) 

                # if mode == "node-max-neuron":
                    # # check if the walk is already in the top walks list
                    # node_level_walk = transform_global_neuron_to_node_neuron(tmp_walk, transforms_modified)
                    # node_level_walk = [node_neuron[0] for node_neuron in node_level_walk]
                    # if node_level_walk in top_k_max_walks: continue

                rel = walk_rel(transforms_modified, init_rel, tmp_walk, mode=mode)

                for i in range(len(sub_walk), len(tmp_walk)+1):
                    sub_walk_rel_dict[tuple(tmp_walk[:i])] = rel
                
        tmp_max_walk = None
        tmp_max_rel = -float('inf')

        del sub_walk_rel_dict[tuple(last_max_walk)]

        for walk in sub_walk_rel_dict.keys():
            if len(walk) < len(last_max_walk): continue
            if tmp_max_rel < sub_walk_rel_dict[walk]:
                tmp_max_rel = sub_walk_rel_dict[walk]
                tmp_max_walk = walk
        
        if tmp_max_walk is None: break
        if mode == "node-max-neuron":
            node_level_walk = transform_global_neuron_to_node_neuron(tmp_max_walk, transforms_modified)
            node_level_walk = tuple([node_neuron[0] for node_neuron in node_level_walk])
            if node_level_walk not in top_k_max_walks_node:
                top_k_max_walks_node.append(node_level_walk)
        top_k_max_walks.append(tmp_max_walk)
        last_max_walk = tmp_max_walk

        if verbose: time_list.append(time.time() - time_a)
        

    real_top_k_max_walks_rels = []
    real_top_k_min_walks_rels = []
    
    if negative_transition_strategy == "min_offset":
        init_rel = real_init_rel
    if mode == "neuron":
        real_top_k_max_walks_rels = [(walk, walk_rel(transforms, init_rel, walk, mode='neuron')) for walk in top_k_max_walks]
    elif mode == 'node-max-neuron':
        real_top_k_max_walks_rels = [(walk, walk_rel(transforms, init_rel, walk, mode='node')) for walk in top_k_max_walks_node]        
    else:
        for walk in top_k_max_walks:
            rel = walk_rel(transforms, init_rel, walk, mode=mode)
            if split_pos_neg:
                if rel >= 0: real_top_k_max_walks_rels.append((walk, rel))
                else: real_top_k_min_walks_rels.append((walk, rel))
            else:
                real_top_k_max_walks_rels.append((walk, rel))
        
    if verbose: return real_top_k_max_walks_rels, real_top_k_min_walks_rels, time_list
    else: return real_top_k_max_walks_rels, real_top_k_min_walks_rels


def plot_top_k_walks(g, top_walks, transforms, rel, mode="neuron", factor=0.3, figname="imgs/topk_walks.eps", color=None, width=12, dataset="MUTAG", idx=None, compute_rel=True, linewidth=13, node_alpha=1):
    

    shrinking_factor = 11
    
    ### Compute pos (np.Array) and node_labels(List)

    if dataset == "BA-2motif":
        pos = nx.kamada_kawai_layout(nx.Graph([(i,j) for i, j in zip(g.get_adj().numpy().nonzero()[0],g.get_adj().numpy().nonzero()[1])]))

        pos_arr = []
        for item in sorted(pos.items(), key=lambda item: item[0]):
            pos_arr.append(list(item[1]))

        pos = np.array(pos_arr)
        node_labels = None

    elif dataset in ["MUTAG", "Mutagenicity"]:

        node_label_dict = \
        {0:'C',1:'O',2:'Cl',3:'H',4:'N',5:'F',6:'Br',7:'S',8:'P',9:'I',10:'Na',11:'K',12:'Li',13:'Ca'} if dataset == 'Mutagenicity' else \
        {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

        atoms = [node_label_dict[i] for i in g.node_tags]
        molecule = Chem.RWMol()
        for atom in atoms:
            molecule.AddAtom(Chem.Atom(atom))
        A = g.get_adj().nonzero()

        for x, y in A:
            if x < y:
                molecule.AddBond(int(x), int(y), Chem.rdchem.BondType.SINGLE)

        AllChem.Compute2DCoords(molecule)
        # compute 2D positions
        pos = []
        n_nodes = molecule.GetNumAtoms()
        for i in range(n_nodes):
            conformer_pos = molecule.GetConformer().GetAtomPosition(i)
            pos.append([conformer_pos.x, conformer_pos.y])
            
        pos = np.array(pos)
        node_labels = atoms

    elif dataset == "Graph-SST2":
        edge_index = np.genfromtxt("datasets/Graph-SST2/raw/Graph_SST2_edge_index.txt")
        node_indicator = np.genfromtxt("datasets/Graph-SST2/raw/Graph_SST2_node_indicator.txt")
        
        nodes = np.where(node_indicator-1 == idx)[0].tolist()
        edges = []
        found = False
        for edge in edge_index:
            if edge[0] in nodes:
                found = True
                edges.append(edge.astype(int).tolist())
            else:
                if found == True:
                    break

        edges = (np.array(edges) - min(nodes)).tolist()
        
        with open("datasets/Graph-SST2/raw/Graph_SST2_sentence_tokens.json", "r") as f:
            token_json = json.load(f)

        tokens = token_json[str(idx)]
        for i in range(len(tokens)):
            if tokens.count(tokens[i]) != 1:
                tokens[i] = tokens[i]+"_"+str(i)
                
        G = nx.DiGraph()
        for n in range(len(g.node_features)):
            G.add_node(tokens[n])

        for edge in edges:
            G.add_edge(tokens[edge[0]], tokens[edge[1]])
            
        pos = np.array(list(nx.drawing.nx_pydot.graphviz_layout(G, prog='dot').values()))  
        node_labels = tokens

    elif dataset == "REDDIT-BINARY":
        # TODO: implement pos and label computation for RB
        raise NotImplementedError("REDDIT-BINARY not implemented.")
    else:
        raise NotImplementedError(datset+" not implemented.")


    ### Compute top walks' scores for visualizing
    if compute_rel == False:
        walk_relevances = top_walks
    else:
        walk_score = {}
        for walk_ in top_walks:
            walk = torch.tensor(walk_[0])
            if mode == "neuron":
                w = tuple(np.array(transform_global_neuron_to_node_neuron(walk, transforms))[:,0])
                if w not in walk_score:
                    walk_score[w] = walk_rel(transforms, rel, w, mode="node")
            else:
                walk_score[walk] = walk_rel(transforms, rel, walk, mode="node")

        walk_relevances = []
        walk_score = sorted(walk_score.items(), key=lambda item: item[1])
        
        for walk, score in walk_score:
            walk_relevances.append((walk, score/max(abs(walk_score[-1][1]), abs(walk_score[0][1]))))

    # plotting
    fig_width = width
    pos_size = pos.max(axis=0) - pos.min(axis=0)
    fig_height = (width / pos_size[0]) * pos_size[1]
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(1, 1, 1)

    def _iterate_over_all_walks(ax, relevances, color=None, linewidth=13):

        # visualization settings
        selfloopwidth = 0.25
        # start iteration over walks
        for walk_id, (walk, relevance) in enumerate(relevances):
            # get walk color
            color = 'b' if relevance < 0 else 'r'
            # get opacity
            alpha = abs(relevance * factor)
            if alpha >1:
                alpha = 1.

            # split position vector in x and y part
            rx = np.array([pos[node][0] for node in walk])
            ry = np.array([pos[node][1] for node in walk])
            
            rx, ry = shrink(rx, ry, shrinking_factor)
            ax.plot(rx, ry, color=color, alpha=alpha, lw=linewidth, zorder=1.)
        return ax

    # plot walks
    if walk_relevances is not None:
        ax = _iterate_over_all_walks(ax, walk_relevances, color, linewidth=linewidth)

    G = nx.from_numpy_matrix(g.get_adj().numpy().astype(int)-np.eye(g.get_adj().shape[0]))
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", alpha=0 if dataset == 'Graph-SST2' else node_alpha, node_size=500)
    collection.set_zorder(2.)
    # plot bonds
    nx.draw_networkx_edges(G, pos, width=2,
        style="dotted" if dataset in ["MUTAG", "Mutagenicity"] else "-")
    # nx.draw(
    #     G,
    #     pos=pos,
    #     with_labels=False,
    #     node_color="k" if node_labels is None else "w",
    #     ,
    #     node_size=300
    # )

    if node_labels is not None:
        pos_labels = pos
        nx.draw_networkx_labels(G, pos_labels, {i: name.split('_')[0] for i, name in enumerate(node_labels)}, font_size=10)

    if dataset == "Graph-SST2":
        # plt.title(" ".join([token.split('_')[0] for token in node_labels]))
        print(" ".join([token.split('_')[0] for token in node_labels]))
    
    # plt.show()
    plt.axis('off')
    
    if figname is not None:
        plt.savefig(figname, dpi=600, format='svg',bbox_inches='tight',  transparent=True)
        # plt.savefig(figname, dpi=600, format='png',bbox_inches='tight',  transparent=True)
    # else:
    #     plt.show()
        
