from typing import List
import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from data_structure import Graph
import pickle as pkl
import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader


def load_ba2motif(dir) -> List[Graph]:
    with open(dir,'rb') as fin:
        adjs, _, labels = pkl.load(fin)
    g_list = []
    for adj, label in zip(adjs, labels):
        edges = np.argwhere(adj==1).tolist()
        g_list.append(Graph(len(adj), edges, int(label[1]), None, None))
    return g_list

def load_mutag(dir):
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open(dir, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            
            g_list.append(Graph(len(g), g.edges, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.edges]
        edges.extend([[i, j] for j, i in edges])

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    # print('# classes: %d' % len(label_dict))
    # print('# maximum node tag: %d' % len(tagset))

    # print("# data: %d" % len(g_list))

    return g_list

def load_mutagenicity(dir):
    node_label_dict = {0:'C',1:'O',2:'Cl',3:'H',4:'N',5:'F',6:'Br',7:'S',8:'P',9:'I',10:'Na',11:'K',12:'Li',13:'Ca'}
    atom_label_dict = {'C':6, 'O':8, 'Cl':17, 'H':1, 'N':7, 'F':9, 'Br':35, 'S':16, 'P':15, 'I':53, 'Na':11, 'K':19, 'Li':3, 'Ca':20}
    amt_node_labels = len(node_label_dict)

    # From PGExplainer
    pri = dir+'_'

    file_edges = pri+'A.txt'
    file_edge_labels = pri+'edge_labels.txt'
    file_edge_labels = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    try:
        edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge label 0')
        edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)

    try:
        node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use node label 0')
        node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1
    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    for (s,t),l in list(zip(edges,edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        if gid !=  graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append([s-start,t-start])
        edge_label_list.append(l)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid!=graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    # return edge_lists, graph_labels, edge_label_lists, node_label_lists

    num_classes = 2
    graphs = []

    for i in range(len(graph_labels)):
        label = graph_labels[i]

        node_tags = node_label_lists[i]

        H_idx = []
        for j, atom_idx in enumerate(node_tags):
            if atom_idx == 3: # H: 3
                H_idx += [j]
        
        non_H_idx = list(range(len(node_tags)))
        for j in H_idx: non_H_idx.remove(j)
        idx_map = dict(np.array([non_H_idx, list(range(len(non_H_idx)))]).T)

        node_tags = np.array(node_tags)[non_H_idx]
        node_features = torch.zeros([len(node_tags),amt_node_labels])
        for j,a in enumerate(node_tags):
            if j in idx_map:
                node_features[idx_map[j]][a] = 1
        edges = []
        for aa, bb in edge_lists[i]:
            if aa in idx_map and bb in idx_map:
                edges.append([aa,bb])

        node_features = np.delete(node_features, 3, axis=1)
        # if adj.shape[0] == node_features.shape[0]:
        graphs.append(Graph(len(idx_map), edges, label, node_tags, node_features))
    

    return graphs

def load_reddit(dir):
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open(dir, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            
            g_list.append(Graph(len(g), g.edges, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.edges]
        edges.extend([[i, j] for j, i in edges])

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    # print('# classes: %d' % len(label_dict))
    # print('# maximum node tag: %d' % len(tagset))

    # print("# data: %d" % len(g_list))

    return g_list

def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data

def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)

def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement

class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
              = read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


def load_graph_sst2(dir):
    dataset = SentiGraphDataset(root=dir, name='Graph_SST2')
    graphs = []
    for i in range(len(dataset)):
        graphs.append(Graph(len(dataset[i].x), dataset[i].edge_index.T.tolist(), dataset[i].y[0], None, dataset[i].x))
    del dataset
    return graphs

def load_collab(dir='datasets/COLLAB'):
    from torch_geometric.datasets import TUDataset
    collab = TUDataset(root=dir, name='COLLAB')
    graphs = []
    for i in range(5000):
        collab_sample = collab.get(i)
        nbnodes = collab_sample.num_nodes
        label = int(collab_sample.y)
        edges = collab_sample.edge_index.numpy().T
        graphs.append(Graph(nbnodes, edges, label, None, None))
        
    return graphs

def load_data(dataset):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    if dataset == 'BA-2motif': 
        g_list = load_ba2motif('datasets/BA_2motif/BA-2motif.pkl')
    elif dataset == 'MUTAG':
        g_list = load_mutag('datasets/MUTAG/MUTAG.txt')
    elif dataset == 'Mutagenicity':
        g_list = load_mutagenicity('datasets/'+dataset+'/'+dataset)
    elif dataset == 'REDDIT-BINARY':
        g_list = load_reddit('datasets/REDDIT_BINARY/reddit_binary.txt')
    elif dataset == 'Graph-SST2':
        g_list = load_graph_sst2('datasets/Graph-SST2')
    elif dataset == 'COLLAB':
        g_list = load_collab()

    print(f"dataset: {dataset}, num_graphs: {len(g_list)}")
    pos_idx = []
    neg_idx = []
    for i, g in enumerate(g_list):
        if g.label == 0:
            pos_idx.append(i)
        else:
            neg_idx.append(i)
    
    return g_list, pos_idx, neg_idx
