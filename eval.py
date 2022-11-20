import pyrootutils

root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torch
import os 
import numpy as np
import warnings
from src.datamodules.components.d2d_dataset import D2DDataset, collate
from src.models.TGT_module import TGTLitModule
from src.models.components.TGT import TGT
from src.models.components.utils.metrics import SumRate
from torch.utils.data import DataLoader
from src.utils.baseline import WMMSEBenchmark, max_power

warnings.filterwarnings("ignore")
all_node_cnt = 0
node_list = [20,30,40,50]
noise = 2.6e-5

for node_num in node_list:
    data_name = 'Node_' + str(node_num)
    data_path = root.as_posix() + '/data/test/' + data_name +'.npy'
    entry_name = str(node_num) + '_' + str(noise).format('.2e')
    path_losses = np.load(data_path, mmap_mode='r')
    alpha = np.ones((path_losses.shape[0],path_losses.shape[1]))
    sr_w= WMMSEBenchmark(noise, path_losses,alpha = alpha)
    sr_one = max_power(noise, path_losses,alpha)
    curr_node_cnt = path_losses.shape[0]
    sr_one_m = np.sum(sr_one) / curr_node_cnt
    sr_w_m = np.sum(sr_w) / curr_node_cnt
    print(f'{node_num} links,{noise:.2e}: Max Power: {sr_one_m:.3f}, WMMSE: {sr_w_m:.3f}')


checkpoint_path = root.as_posix() + '/model_checkpoints'
ck_list = []
for file in os.listdir(checkpoint_path):
    if file.endswith('.ckpt'):
        file_name = file.split('.c')[0]
        ck_list.append([file_name, os.path.join(checkpoint_path, file)])

node_list = [20,30,40,50]
noise = 2.6e-5
for name, ck in ck_list:
    checkpoint = torch.load(ck, map_location=lambda storage, loc: storage)
    model = TGT()
    litmodel = TGTLitModule.load_from_checkpoint(ck,net=model)
    for node_num in node_list:
        data_path = root.as_posix() + '/data/test/Node_' + str(node_num) +'.npy'
        path_losses = np.load(data_path, mmap_mode='r')
        testset = D2DDataset(path_losses)
        testloader = DataLoader(testset, batch_size=16, shuffle=False, collate_fn=collate)
        srate = SumRate()
        allrate = 0
        for g in testloader:
            if name.startswith('UW'):
                allocs = litmodel.net(g,var=torch.tensor(noise).float())
                srate(g, allocs,noise)
            elif name.startswith('Weights'):
                weights = torch.rand_like(g.ndata['feat'][:,1])
                allocs = litmodel.forward(g,weights)
                srate(g, allocs, noise, weights)
            else:
                allocs = litmodel.forward(g)
                srate(g, allocs,noise)
        allrate = srate.compute()
        print(f'{name} with {node_num} pairs,{noise:.2e}: {allrate:.3f}')
        