from two_stage_deferral import Deferral_Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import h5py
import numpy as np

def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        action="store",
        default=0,
        type=int,
        help="Random seed for pl.seed_everything function.",
    )
    
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        action="store",
        default=64,
        type=int,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--alpha",
        dest="alpha",
        action="store",
        default=0.5,
        type=float,
        help="Deferral cost",
    )

    args = parser.parse_args()

    git_hash = get_git_revision_short_hash()
    human_readable_extra = ""
    experiment_name = "-".join(
        [
            git_hash,
            f"seed={args.random_seed}",
            "liu",
            str(args.alpha),
            human_readable_extra,
            f"bs={args.batch_size}",
        ]
    )

    pl.seed_everything(seed=args.random_seed)

    hdf5_file = h5py.File("/AIROGS.h5", "r")
    labels_train = hdf5_file["train/diagnosis"][:].tolist()
    labels_val = hdf5_file["val/diagnosis"][:].tolist()

    print(f"Training dataset length: {len(labels_train)}")
    print(f"Validation dataset length: {len(labels_val)}")

    def get_oversampler(dataset):
        # oversample minority class (RG)
        labels = dataset.tensors[1].long()
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return sampler
    
    # make datasets for dataloader
    x = np.load("/LD2_input.npz")
    x_train = x["train"].tolist()
    x_val = x["val"].tolist()

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(labels_train, dtype=torch.long,))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(labels_val, dtype=torch.long,))


    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=False, sampler=get_oversampler(train_dataset)
        )
    valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, drop_last=False, sampler=get_oversampler(valid_dataset)
    )

    logger = TensorBoardLogger(
        save_dir="./runs", name=experiment_name, default_hp_metric=False
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            filename="best-loss-{epoch}-{step}",
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val/pauc",
            filename="best-pauc-{epoch}-{step}",
            mode="max",
        )
    ]
    
    model = Deferral_Model(22, 3, alpha = args.alpha)
   
    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.5,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        callbacks=checkpoint_callbacks,
        max_epochs=5,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )