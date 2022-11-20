import torch
import torch.nn as nn 
from torch.functional import F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid


class Embedding(nn.Module):
    def __init__(self,node_dim, e_dim, nemb_dim, eemb_dim):
        super(Embedding, self).__init__()
        self.nlin = Sequential(Linear(node_dim, nemb_dim), ReLU(), BatchNorm1d(nemb_dim,track_running_stats=False))
        self.elin = Sequential(Linear(e_dim, eemb_dim), ReLU(), BatchNorm1d(eemb_dim,track_running_stats=False))

    def forward(self, n_feat, e_feat):
        n_emb = self.nlin(n_feat)
        e_emb = self.elin(e_feat)
        return n_emb, e_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
            
        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)

        
    def split_heads(self,x):
        return x.view(-1, self.num_heads, self.d_k)
    
    def group_heads(self,x):
        seq_len = x.shape[1]
        return x.view(-1,seq_len, self.d_model)
    

    def forward(self, g, q,e,k,v):
        # After transforming, split into num_heads 
        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = self.split_heads(self.W_v(v))
        g.edata['e'] = e

        g.update_all(self.message_func, self.reduce_func)
        return g.ndata['h']

    def message_func(self,edges):
        X_q = torch.cat([edges.dst['q']], dim=-1)
        X_k = torch.cat([edges.src['k']], dim=-1)

        X_k = X_k  + edges.data['e']
        K = self.split_heads(self.W_k(X_k))
        Q = self.split_heads(self.W_q(X_q))

        scores = torch.matmul(Q, K.transpose(1,2)) / torch.sqrt(torch.tensor(self.d_k))
        scores = F.leaky_relu(scores)
        return {'V': edges.src['v'], 's': scores}

    def reduce_func(self,nodes):
        A = F.softmax(nodes.mailbox['s'], dim=-1) 
        H = torch.matmul(A , nodes.mailbox['V'])
        H = self.group_heads(H)
        H = torch.sum(H,dim=1)
        return {'h': H}


class TGTLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,d_input = (d_model , d_model , d_model))
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self,g, x, e):

        attn_output= self.mha(g ,x, e, x, x)
        out = self.layernorm1(x + attn_output) 

        return out

class TGT(nn.Module):
    def __init__(self, d_model=64, num_heads=32, num_layers = 3):
        super().__init__()
        num_heads = d_model//2
        self.num_layers = num_layers
        self.shared_mha = TGTLayer(d_model, num_heads) 
        self.enc_layers = nn.ModuleList([
            self.shared_mha for i in range(self.num_layers)])
        self.emb = Embedding(2,2, d_model, d_model)

    def forward(self, g):
        x, e = self.emb(g.ndata['feat'], g.edata['feat'])

        for i in range(self.num_layers):
            x= self.enc_layers[i](g, x, e)

        p_opt = Sigmoid()(torch.sum(x,dim=-1).unsqueeze(-1))
        return p_opt
