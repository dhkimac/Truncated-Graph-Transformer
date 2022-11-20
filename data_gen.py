from typing import Any, Dict, Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.datamodules.d2d_datamodule import D2DDataModule

node_num = [[20],[30],[40],[50]]
for n in node_num:
    datamodule = D2DDataModule(data_dir= "/data/train/", data_name='Node_',train_val_test_split=(25000, 2500, 200),n_list = n)
    datamodule.prepare_data(new_data=True)

for n in node_num:
    datamodule = D2DDataModule(data_dir= "/data/test/", data_name='Node_',network_size = 50, num_channels=20,n_list = n)
    datamodule.prepare_data(new_data=True)