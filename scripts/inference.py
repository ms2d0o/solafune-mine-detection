import os
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import timm
import torch
from torch.utils.data import DataLoader

from dataset import MineDataset
from steps import inference

@hydra.main(version_base="1.2", config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    print(cfg)

    print("number of experiments: ", len(cfg.experiments))
    df_test = pd.read_csv(
        'data/uploadsample.csv',
        header=None,
        names=['filename', 'target'])
    test_ds = MineDataset(cfg.INPUT_ROOT, df_test, transform=[])
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    if cfg.device == "mps" and torch.backends.mps.is_built():
        device = "mps"
        print("Inferece on MPS")
    elif cfg.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print("Inferece on GPU")
    else:
        device = "cpu"
        print("Inferece on CPU")

    prediction = None
    for i, exp in enumerate(cfg.experiments):
        model_paths = Path(os.path.sep.join([cfg.model_dir,exp.ID])).glob("**/*.pth")
        print(f"Experiment: {i+1}/{len(cfg.experiments)}")
        for p in model_paths:
            model = timm.create_model(exp.model, pretrained=False, num_classes=2, in_chans=exp.in_chans).to(device)
            model.load_state_dict(torch.load(p, map_location=device))
            preds = inference(model, test_dl,device)
            if prediction is None:
                prediction = torch.cat(preds['pred'], dim=0).cpu().numpy()
            else:
                prediction += torch.cat(preds['pred'], dim=0).cpu().numpy()
    predict_labels = prediction.argmax(1)
    df_test['target'] = predict_labels
    # now datetime to str
    now_str = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    df_test.to_csv(cfg.OUTPUT_ROOT+f"/submission_{now_str}.csv", index=False, header=False)

if __name__ == "__main__":
    main()