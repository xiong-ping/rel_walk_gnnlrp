from re import L
from sklearn.metrics import accuracy_score, roc_auc_score
from subgraph_relevance import subgraph_mp_transcription
import numpy as np
import torch
from rdkit.Chem import AllChem
import networkx as nx
from rdkit import Chem
from matplotlib import pyplot as plt

def shrink(rx, ry, factor=11):
    """This function is used to make the walks smooth."""

    rx = np.array(rx)
    ry = np.array(ry)

    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    last_node = rx.shape[0] - 1
    concat_list_x = [np.linspace(rx[0], rx[0], 5)]
    concat_list_y = [np.linspace(ry[0], ry[0], 5)]
    for j in range(last_node):
        concat_list_x.append(np.linspace(rx[j], rx[j + 1], 5))
        concat_list_y.append(np.linspace(ry[j], ry[j + 1], 5))
    concat_list_x.append(np.linspace(rx[last_node], rx[last_node], 5))
    concat_list_y.append(np.linspace(ry[last_node], ry[last_node], 5))

    rx = np.concatenate(concat_list_x)
    ry = np.concatenate(concat_list_y)

    filt = np.exp(-np.linspace(-2, 2, factor) ** 2)
    filt = filt / filt.sum()

    rx = np.convolve(rx, filt, mode='valid')
    ry = np.convolve(ry, filt, mode='valid')

    return rx, ry

def plot_mutagenicity(g, relevances=None, width=12, shrinking_factor=11, factor=1, color=None, figname=None, dataset='Mutagenicity', node_label_dict=None):
    """plot the molecular, optional with relevances

    Args:
        relevances (List, optional): [[walk], relevance], like [[1,2,3],0.3] means the walk 1,2,3 has relevance 0.3. Defaults to None.
        width (int, optional): figure width. Defaults to 12.
        shrinking_factor (int, optional): shrink factor used for smoothing walk. Defaults to 11.
        factor (int, optional): multiply with the relevance score to make plot prettier. Defaults to 1.

    Returns:
        [type]: [description]
    """   

    if node_label_dict is None:
        if dataset == 'Mutagenicity':
            node_label_dict = {0:'C',1:'O',2:'Cl',3:'H',4:'N',5:'F',6:'Br',7:'S',8:'P',9:'I',10:'Na',11:'K',12:'Li',13:'Ca'}
        else:            
            node_label_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    
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

    # plotting
    fig_width = width
    pos_size = pos.max(axis=0) - pos.min(axis=0)
    fig_height = (width / pos_size[0]) * pos_size[1]
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(1, 1, 1)

    ####################################################################################################################
    # Utils
    ####################################################################################################################

    def _iterate_over_all_walks(ax, relevances, color=None):

        # visualization settings
        selfloopwidth = 0.25
        linewidth = 13.
        # start iteration over walks
        for walk_id, (walk, relevance) in enumerate(relevances):
            # get walk color
            if color is None:
                color = 'b' if relevance < 0 else 'r'
            # get opacity
            alpha = abs(relevance * factor)
            if alpha >1:
                alpha = 1.

            # split position vector in x and y part
            rx = np.array([pos[node][0] for node in walk])
            ry = np.array([pos[node][1] for node in walk])
            # plot g loops
            for i in range(len(rx) - 1):
                if rx[i] == rx[i + 1] and ry[i] == ry[i + 1]:
                    rx_tmp = rx[i] + selfloopwidth * np.cos(np.linspace(0, 2 * np.pi, 128))
                    ry_tmp = ry[i] + selfloopwidth * np.sin(np.linspace(0, 2 * np.pi, 128))
                    ax.plot(rx_tmp, ry_tmp, color=color, alpha=alpha, lw=linewidth, zorder=1.)
            # plot walks
            rx, ry = shrink(rx, ry, shrinking_factor)
            ax.plot(rx, ry, color=color, alpha=alpha, lw=linewidth, zorder=1.)
        return ax

    ####################################################################################################################
    # Main function code
    ####################################################################################################################

    # plot walks
    if relevances is not None:
        ax = _iterate_over_all_walks(ax, relevances, color)
    # ax = _iterate_over_all_walks(ax, [([0, 1, 23, 25], 0.09783325320052568)])

    G = nx.from_numpy_matrix(g.get_adj().numpy().astype(int)-np.eye(g.get_adj().shape[0]))
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", alpha=0, node_size=500)
    collection.set_zorder(2.)
    # plot bonds
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color="w",
        width=4,
        style="dotted",
        node_size=300
    )
    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    nx.draw_networkx_labels(G, pos_labels, {i: name for i, name in enumerate(atoms)}, font_size=30)

    plt.axis('off')
    plt.show()

    if figname is not None:
        plt.savefig(figname, dpi=600, format='eps',bbox_inches='tight')
        

def plot_synthetic(g, S=None, walks=None, node_size=30, width=12, factor=1, shrinking_factor=11):
    G = nx.from_numpy_matrix(g.get_adj().numpy()-np.eye(g.nbnodes))
    pos = nx.spring_layout(G, seed=10)
    
    color_map = []
    alpha_map = []


    if S is None: S = []
    else:
        subgraph_edges = nx.subgraph(G, S).edges
        nx.draw_networkx_edges(G, pos, edgelist=subgraph_edges, alpha=1)

    if walks is None: 
        walks = []
    else:
        S = list(set(np.array(walks).flatten().tolist()))

    for node in G:
        if node in S:
            color_map.append('grey')
            alpha_map.append(0.4)
        else: 
            color_map.append('grey')
            alpha_map.append(0.4)

    
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_size, alpha=alpha_map)
    nx.draw_networkx_edges(G, pos, alpha=0.1)

    
    
    for walk in walks:
        nx.draw_networkx_edges(G, pos, edgelist=[[walk[i], walk[i+1]] for i in range(len(walk) - 1)], alpha=1, edge_color='r')
    
    plt.savefig('imgs/topk_walks.svg', dpi=600, format='svg',bbox_inches='tight')
    plt.show()
