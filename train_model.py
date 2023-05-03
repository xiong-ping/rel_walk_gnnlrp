import torch
import modules
from tqdm import tqdm
import numpy as np
import random

def train_model(dataset, train_idx, test_idx, config):
    # config = {
    #     'num_layer': 8,
    #     'mode': 'gin',
    #     'epochs': 1000,
    #     'lr': 0.00001,
    #     'model_dir': 'models/gin-8-ba2motif.torch',
    #     'nbclasses': 2,
    #     'inter_feat_dim': 20
    # }
    num_layer= config['num_layer']
    mode = config['mode']
    epochs = config['epochs']
    lr = config['lr'] 
    model_dir = config['model_dir']
    nbclasses = config['nbclasses']
    inter_feat_dim = config['inter_feat_dim']
    print_out_nb = config['print_out_nb']
    optimizer_label = config['optimizer']

    H0_dim = dataset[0].node_features.shape[1] if dataset[0].node_features != None else 1

    print('train {}'.format(model_dir))
    if mode == 'gcn': nn = modules.GNN([[H0_dim],[H0_dim,inter_feat_dim]] +
                                        [[inter_feat_dim,inter_feat_dim] ]*(num_layer-2)+ 
                                        [[inter_feat_dim,nbclasses]],mode='std')
    if mode == 'gin': nn = modules.GNN([[H0_dim],[H0_dim,inter_feat_dim,inter_feat_dim]]+
                                        [[inter_feat_dim,inter_feat_dim,inter_feat_dim]]*(num_layer -2)+
                                        [[inter_feat_dim,nbclasses]],mode='std')

    optimizer = torch.optim.SGD(nn.params, lr=lr, momentum=0.9) if optimizer_label == 'sgd' else torch.optim.Adam(nn.params, lr=lr)
    loss = torch.nn.BCEWithLogitsLoss()
    print('{}: epoch loss errors'.format(mode))
    for epoch in tqdm(range(1,epochs +1)):
        random.shuffle(train_idx)
        for it, sample_num in enumerate(train_idx):
            if optimizer_label == 'sgd':
                for g in optimizer.param_groups: g['lr'] = lr / (1.0 + (epoch / epochs))

            optimizer.zero_grad()
            g = dataset[sample_num]
            y = nn.forward(g.get_adj(), H0=g.node_features)
            t = g.label
            obj = loss(y.unsqueeze(0),torch.eye(nbclasses)[t:t+1])
            obj.backward()
            optimizer.step()

        if epoch%print_out_nb==0:
            err = 0
            losses = np.zeros([len(test_idx)])
            errors = np.zeros([len(test_idx)])
            for i, test_sample in enumerate(test_idx):
                g = dataset[test_sample]
                y = nn.forward(g.get_adj(), H0=g.node_features).data
                t = g.label
                losses[i] = loss(y.unsqueeze(0),
                                torch.eye(nbclasses)[t:t+1]).data.numpy()

                errors[i] = ((y.argmax() != t)*1.0).data.numpy()

            print('% 8d %.3f %.3f'%(epoch,losses.mean(),errors.mean()))
        torch.save(nn,model_dir)