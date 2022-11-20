import numpy as np
import torch


def generate_layouts( nLayouts, nNodes, r=1, pl=2.2 ,min_dist=1):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(nLayouts, nNodes))
    while(True):
        transmitters = np.random.uniform(low=-nNodes/r, high=nNodes/r, size=(nLayouts,nNodes,2))
        pair_dist = np.random.uniform(low=min_dist, high=nNodes/4, size =(nLayouts,nNodes))
        pair_angles = np.random.uniform(low=0, high=np.pi*2, size =(nLayouts,nNodes))
        pair_x =  pair_dist * np.cos(pair_angles)
        pair_y =  pair_dist * np.sin(pair_angles)
        receivers = transmitters + np.stack([pair_x,pair_y],axis=-1)
        transmitters = torch.tensor(transmitters, dtype=torch.float32)
        receivers = torch.tensor(receivers, dtype=torch.float32)
        distances = torch.cdist(transmitters, receivers)
        path_gain = torch.pow(distances,-pl).numpy()
        if not np.any(np.isinf(path_gain)):
            return( dict(zip(['tx', 'rx'],[transmitters, receivers] )), path_gain )
        else:
            print("inf found, trying again")

def compute_fading(path_gains, gamma=1.):
    samples = np.random.rayleigh(gamma, (path_gains.shape))
    channel = samples * path_gains
    return channel

