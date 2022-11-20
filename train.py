import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from functools import partial
import torch
from pytorch_lightning import Trainer
from src.datamodules.d2d_datamodule import D2DDataModule
from src.models.components.TGT import TGT
from src.models.TGT_module import TGTLitModule
node_num = [[20],[30],[40],[50],[20,30,40,50]]
noise = 2.6e-5
for n in node_num:
    model = TGT(64,32)
    run_name = f"TGT_{n}"
    datamodule = D2DDataModule(data_name='Node_',train_val_test_split=(25000, 2500, 200),n_list = n,batch_size=64)
    optimizer = partial(torch.optim.AdamW, lr=5e-4)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, mode="min", patience=10, verbose=True)
    lit_module = TGTLitModule(model, optimizer, scheduler, noise)
    trainer = Trainer(max_epochs=50,
            accelerator='gpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=5.0
        )
    trainer.fit(lit_module, datamodule=datamodule)
    trainer.save_checkpoint(root.as_posix() + '/model_checkpoints/' + run_name + '.ckpt')