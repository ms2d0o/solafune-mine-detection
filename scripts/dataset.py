from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from albumentations.pytorch import ToTensorV2
import albumentations as A


def get_true_color(raster) -> np.ndarray:
    return raster.read([1, 2, 3])


class MineDataset(Dataset):
    def __init__(self, imagedir: str | Path, df: pd.DataFrame, transform: list | None = None):
        self.df = df
        self.imagedir = Path(imagedir)
        self.transform = transform
        if self.transform is None:
            self.transform = []
        self.transform.append(
            ToTensorV2(always_apply=True)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'filename']
        filepath = (self.imagedir / filename).with_suffix('.npy')
        image = np.load(filepath)

        if self.transform:
            img_tensor = A.Compose(self.transform)(image=image)['image']
        return (img_tensor, self.df.loc[idx, 'target'])
