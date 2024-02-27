from tqdm import tqdm
import torch
from collections import OrderedDict, defaultdict
import numpy as np


def train_step(model, dataloader, criterion, optimizer, scheduler, device):
    # train mode
    model.train()

    train_losses = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="[Train]")
    for step, (x, y) in pbar:
        # x: torch.Tensor, shape=(batch_size, 3, image_size, image_size)
        # y: torch.Tensor, shape=(batch_size, )
        x, y = x.float().to(device), y.to(device)
        torch.set_grad_enabled(True)
        pred = model(x)
        loss = criterion(pred, y)
        train_losses.append(loss.item())

        pbar.set_postfix(
            OrderedDict(
                loss=np.mean(train_losses),
            )
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    train_loss = np.mean(train_losses)
    return train_loss


def test_step(model, dataloader, criterion, device):
    # eval
    model.eval()
    test_losses = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="[Test]")
    with torch.no_grad():
        for step, (x, y) in pbar:
            x, y = x.float().to(device), y.to(device)
            pred = model(x).softmax(1)
            loss = criterion(pred, y)
            test_losses.append(loss.item())

            pbar.set_postfix(
                OrderedDict(
                    loss=np.mean(test_losses),
                )
            )
    test_loss = np.mean(test_losses)
    return test_loss


def inference(model, dataloader, device):
    # eval
    model.eval()
    torch.set_grad_enabled(False)

    preds = defaultdict(list)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="[Inference]")
    with torch.no_grad():
        for step, (x, y) in pbar:
            x, y = x.float().to(device), y.to(device)
            pred = model(x).softmax(1)
            preds['pred'].append(pred)
            preds['y'].append(y)

    return preds
