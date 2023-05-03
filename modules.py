
import torch
import numpy
import scipy,scipy.linalg

train = False

def softmin(b): return -0.5*torch.log(1.0+torch.exp(-2*b))

class Block:
    
    def __init__(self,s,k):
        self.W = []
        self.B = []
        for a,b in zip(s[:-1],s[1:]):
            self.W += [
                torch.nn.Parameter(torch.FloatTensor([
                    numpy.random.normal(0,a**-.5,[a,b]) for _ in range(k)
                ]))
            ]
            self.B += [
                torch.nn.Parameter(torch.FloatTensor([
                    numpy.random.normal(0,1,[b]) for _ in range(k)
                ]))
            ]

    def forward(self,Hin,A,mask=1):

        n = Hin.shape[0]

        for Wo,Bo,Ao in zip(self.W,self.B,[A*mask]+[torch.eye(n).reshape(1,n,n)]*(len(self.W)-1)):

            Hout = 0
            for ao,wo,bo in zip(Ao,Wo,Bo):
                bo = softmin(bo)
                Hout = Hout + ao.permute(1,0).matmul(Hin).matmul(wo) + bo

            if Wo.shape[2] > 10:
                Hin = Hout.clamp(min=0)
            else:
                Hin = Hout

        return Hin

    def lrpforward(self,Hin,A,gamma):

        n = Hin.shape[0]

        for Wo,Bo,Ao in zip(self.W,self.B,[A]+[torch.eye(n).reshape(1,n,n)]*(len(self.W)-1)):


            Hout = 0
            Pout = 1e-6
            for ao,wo,bo in zip(Ao,Wo,Bo):
                bo = softmin(bo)
                Hout = Hout + ao.permute(1,0).matmul(Hin).matmul(wo) + bo

                if gamma > 0 and wo.shape[-1] > 10:  
                
                    wp = wo + gamma*wo.clamp(min=0)
                    bp = bo + gamma*bo.clamp(min=0)
                    Pout = Pout + ao.permute(1,0).matmul(Hin).matmul(wp) + bp

            if gamma > 0 and wo.shape[-1] > 10:  
                Hout = Pout * (Hout / Pout).data

            if Wo.shape[2] > 10:
                Hin = Hout.clamp(min=0)
            else:
                Hin = Hout

        return Hin

class GNN:

    def __init__(self,sizes,mode='std'):
        if mode == 'std': k=1
        if mode == 'cheb': k=3

        self.d = sizes[0][0]

        self.blocks = [Block(s,k) for s in sizes]
        self.mode = mode
        self.params = []
        for l in self.blocks: self.params += l.W

    def adj(self,A):

        if self.mode == 'std':
            L1 = A / 2
            A = torch.cat((L1.unsqueeze(0),)) 
            return A/1**.5

        if self.mode == 'cheb':
            L0 = torch.eye(len(A))
            L1 = A / 2
            L2 = L1.matmul(L1)
            A = torch.cat((L0.unsqueeze(0),L1.unsqueeze(0),L2.unsqueeze(0)))
            return A/3**.5

    def ini(self,A,H0):
        if H0 == None: H0 = torch.ones([len(A),1])
        return H0

    def forward(self,A,H0=None,masks=None):
        
        if masks is None:
            masks = [1]*(len(self.blocks)-1)
        H0 = self.ini(A, H0)

        H = self.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0))

        A = self.adj(A)

        for l,mask in zip(self.blocks[1:],masks):
            H = l.forward(H,A,mask=mask)

        H = H.sum(dim=0) / 20**.5
        return H

    def lrp(self,A,gammas,t,inds,H0=None):
        
        H0 = self.ini(A, H0)
        H1 = self.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0)).data

        A = self.adj(A)

        H1.requires_grad_(True)

        H = H1

        if inds is None:
            for l,gamma in zip(self.blocks[1:],gammas):
                H = l.lrpforward(H,A,gamma)
        else:
            
            for l,i,gamma in zip(self.blocks[1:],inds,gammas):
                H = l.lrpforward(H,A,gamma)
                M = torch.FloatTensor(numpy.eye(H.shape[0])[i][:,numpy.newaxis])
                H = H * M + (1-M) * (H.data)

        H = H.sum(dim=0) / 20**.5

        H[t].backward()
        return (H1*H1.grad).sum(dim=1).data

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GNNExplainer, GINConv, MessagePassing, GCNConv, GraphConv


class Net1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type):
        super(Net1, self).__init__()
        dim = 32
        self.convs = torch.nn.ModuleList()
        if conv_type == 'GCNConv':
            conv_class = GCNConv
            kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim, **kwargs))
        self.concat_features = concat_features
        if concat_features:
            self.fc = Linear(dim * num_layers + num_node_features, num_classes)
        else:
            self.fc = Linear(dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)