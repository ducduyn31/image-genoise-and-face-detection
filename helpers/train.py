import pytorch_lightning as pl
import pathlib

from custom_dataset.sr_wider_datamodule import SRWiderfaceDataModule
from models.image_logger import ImageLogger
from models.swinir import SwinIR


def train_id_model():
    pl.seed_everything(42, workers=True)
    ROOT = pathlib.Path('../data')
    data_module = SRWiderfaceDataModule(root=ROOT)

    model = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        upscale=8,
        img_range=1.0,
        upsampler='nearest+conv',
        resi_connection='1conv',
    )

    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                every_n_train_steps=10000,
                save_top_k=-1,
                filename='{epoch:02d}-{val_loss:.6f}',
            ),
            ImageLogger(
                log_every_n_steps=1000,
                max_images_each_step=4,
            ),
        ],
        accelerator='ddp',
        precision=32,
        gpus=[0, 1, 2],
        default_root_dir=ROOT / 'logs',
        max_steps=150001,
        val_check_interval=500,
        log_every_n_steps=50,
        accumulate_grad_batches=1,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    train_id_model()
