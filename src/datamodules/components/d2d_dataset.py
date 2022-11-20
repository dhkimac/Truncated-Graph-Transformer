import dgl
from dgl.data import DGLDataset
import torch
import torch.nn.functional as F
import numpy as np

# Generate DGL Dataset from CSI data
class D2DDataset(DGLDataset):
    def __init__(self, data, normalize=True, self_loop=True):
        self.csi = data
        self.norm_csi = F.normalize(torch.tensor(data), p=2, dim=1).numpy()
        self.n = data.shape[-1]
        self.normalize = normalize
        self.self_loop = self_loop
        self.adj = self.generate_adjacency_matrix(self.n)
        super().__init__(name="D2D_SISO")

    def process(self):
        self.graphs = []
        for i in range(len(self.csi)):
            graph = self.build_graph(i)
            self.graphs.append(graph)
    
    def build_graph(self, i):
        graph = dgl.graph(self.adj, num_nodes=self.n)

        if self.normalize:
            H = self.norm_csi[i,:,:]
        else:
            H = self.csi[i,:,:]
        csi = self.csi[i,:,:]

        node_features = torch.tensor(np.expand_dims(np.diag(H),axis=1), dtype = torch.float)
        node_features = torch.cat([node_features, torch.ones_like(node_features)], axis = 1)

        edge_features  = []
        edge_csi = []
        for e in self.adj:
            edge_features.append([H[e[0],e[1]],H[e[1],e[0]]])
            edge_csi.append(csi[e[0],e[1]])
        
        graph.ndata['csi'] = torch.tensor(np.expand_dims(np.diag(csi),axis=1), dtype = torch.float)
        graph.ndata['feat'] = node_features
        graph.edata['feat'] = torch.tensor(edge_features, dtype = torch.float)
        graph.edata['csi'] = torch.tensor(edge_csi, dtype = torch.float)

        return graph

    def generate_adjacency_matrix(self, n):
        adj = []
        for i in range(0,n):
            for j in range(0,n):
                if(self.self_loop or not(i==j)):
                    adj.append([i,j])
        return adj

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

def collate(samples):
    batched_graph = dgl.batch(samples)
    return batched_graph

if __name__ == "__main__":
    data = np.random.randn(10, 3, 3)
    dataset = D2DDataset(data)
    print(dataset[0], len(dataset))