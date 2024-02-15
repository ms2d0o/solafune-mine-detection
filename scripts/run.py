import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import timm
from transformers import get_cosine_schedule_with_warmup

from dataset import MineDataset
from steps import train_step, test_step
from train import training
from torch.utils.data import DataLoader


@hydra.main(version_base="1.2", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.device == "mps" and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    df_dataset = pd.read_csv(
        'data/answer.csv',
        header=None,
        names=['filename', 'target'])

    cv = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.SEED
    )
    list_cv = list(cv.split(df_dataset, df_dataset['target']))

    for i, (train_idx, valid_idx) in enumerate(list_cv):
        print(f"Fold: {i+1}/{cfg.n_splits}")
        df_train = df_dataset.loc[train_idx].reset_index(drop=True)
        df_valid = df_dataset.loc[valid_idx].reset_index(drop=True)

        print(df_train.head())
        print(df_valid.head())

        train_ds = MineDataset(
            imagedir=os.path.sep.join([cfg.INPUT_ROOT, 'train']),
            df=df_train
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            # num_workers=cfg.NUM_WORKERS
        )
        valid_ds = MineDataset(
            imagedir=os.path.sep.join([cfg.INPUT_ROOT, 'train']),
            df=df_valid
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        model = timm.create_model(
            cfg.model.name,
            **cfg.model.params
        ).to(device)
        criterion = getattr(
            torch.nn,
            # "CrossEntropyLoss"
            cfg.criterion.name
        )(**cfg.criterion.params)
        optimizer = getattr(
            torch.optim,
            # "AdamW"
            cfg.optimizers.name
        )(
            model.parameters(),
            **cfg.optimizers.params
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps * len(train_dl),
            num_training_steps=cfg.EPOCHS * len(train_dl)
        )
        training(
            model,
            train_dl,
            valid_dl,
            criterion,
            optimizer,
            scheduler,
            cfg.EPOCHS,
            cfg.OUTPUT_ROOT,
            device
        )


if __name__ == "__main__":
    main()
