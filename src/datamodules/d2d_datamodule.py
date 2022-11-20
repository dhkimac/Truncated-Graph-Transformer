from typing import Any, Dict, Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.datamodules.components.d2d_dataset import D2DDataset, collate
from src.datamodules.components.wireless_networks_generator import generate_layouts, compute_fading     
import numpy as np
import os 

class D2DDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/train/",
        train_val_test_split: Tuple[int, int, int] = (25000,2500,200),
        network_size: int = (500+50+4),
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = False,
        self_loop = True,
        normalize=True,
        num_channels = 50,
        data_name = 'Node_',
        n_list = [20],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self, new_data=False):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        for node_num in self.hparams.n_list:
            name = self.hparams.data_name + str(node_num)
            data_path = root.as_posix() + self.hparams.data_dir + name +'.npy'
            if new_data or not os.path.exists(data_path):
                total_train_data = self.hparams.network_size
                train_data_per_n = total_train_data // len(self.hparams.n_list)

                data_path = root.as_posix() + self.hparams.data_dir + name +'.npy'
                layouts, path_gains = generate_layouts(train_data_per_n,node_num)
                path_gains = np.repeat(path_gains,self.hparams.num_channels,axis=0)
                channels = compute_fading(path_gains)

                np.save(data_path, channels)

        
    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_list = []
            name = self.hparams.data_name 
            for n in self.hparams.n_list:
                data_name = name + str(n)
                data_path = root.as_posix() + self.hparams.data_dir  + data_name +'.npy'
                channels = np.load(data_path, mmap_mode='r')
                trunc_channels = channels[:(self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1] + self.hparams.train_val_test_split[2])//len(self.hparams.n_list)]
                trainset = D2DDataset(trunc_channels,normalize=self.hparams.normalize,self_loop=self.hparams.self_loop)
                dataset_list.append(trainset)

            dataset = ConcatDataset(datasets=dataset_list)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )
        

if __name__ == "__main__":
    dm = D2DDataModule()
    dm.prepare_data()
    dm.setup()
    print(dm.data_train[0])