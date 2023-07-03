import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import os
import warnings
from functools import partial
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import numpy as np
from src.datamodules.components.d2d_dataset import D2DDataset, collate
from src.datamodules.d2d_datamodule import D2DDataModule
from src.models.components.TGT import TGT
from src.models.components.utils.metrics import SumRate
from src.models.TGT_module import TGTLitModule
from src.utils.baseline import WMMSEBenchmark, max_power

#Generate Data
print("Generating Data")
warnings.filterwarnings("ignore")
data_node_num = [[20],[30],[40],[50]]
for n in data_node_num:
    if not os.path.exists(f"./data/train/Node_{n[0]}.npy"):
        datamodule = D2DDataModule(data_dir= "/data/train/", data_name='Node_',train_val_test_split=(25000, 2500, 200),n_list = n)
        datamodule.prepare_data(new_data=True)

for n in data_node_num:
    if not os.path.exists(f"./data/test/Node_{n[0]}.npy"):
        datamodule = D2DDataModule(data_dir= "/data/test/", data_name='Node_',network_size = 50, num_channels=20,n_list = n)
        datamodule.prepare_data(new_data=True)

#Train Model
print("Training Model")
model_node_num = [[30],[50],[20,30,40,50]]
noise = 2.6e-5
model_list = []
for n in model_node_num:
    if os.path.exists(f"./model_checkpoints/TGT_{n}.ckpt"):
        model = TGT(64,32)
        lit_module = TGTLitModule.load_from_checkpoint(f"./model_checkpoints/TGT_{n}.ckpt", net=model)
        model_list.append(lit_module)
        continue

    model = TGT(64,32)
    run_name = f"TGT_{n}"
    datamodule = D2DDataModule(data_name='Node_',train_val_test_split=(25000, 2500, 200),n_list = n,batch_size=64)
    optimizer = partial(torch.optim.AdamW, lr=5e-4)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, mode="min", patience=10, verbose=True)
    lit_module = TGTLitModule(model, optimizer, scheduler, noise)
    model_list.append(lit_module)
    trainer = Trainer(max_epochs=30,
            accelerator='gpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=5.0
        )
    trainer.fit(lit_module, datamodule=datamodule)
    trainer.save_checkpoint(root.as_posix() + '/model_checkpoints/' + run_name + '.ckpt')

#Test Model
print("Testing Model")
all_node_cnt = 0

for node_num in data_node_num:
    data_name = 'Node_' + str(node_num[0])
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
    for i, litmodel in enumerate(model_list):
        testset = D2DDataset(path_losses)
        testloader = DataLoader(testset, batch_size=16, shuffle=False, collate_fn=collate)
        srate = SumRate()
        for g in testloader:
            allocs = litmodel.forward(g)
            srate(g, allocs,noise)
            allrate = srate.compute()
        print(f'TGT trained with {model_node_num[i]} pairs: {allrate:.3f}')
