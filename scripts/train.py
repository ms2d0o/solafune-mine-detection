import numpy as np
import torch
from collections import defaultdict

from steps import train_step, test_step


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training(
        model,
        train_dl,
        valid_dl,
        criterion,
        optimizer,
        scheduler,
        epochs,
        output_dir,
        device
        ):

    training_results = defaultdict(list)
    best_loss = float('inf')

    # TODO: saverのパラメータをconfigから取得する
    for epoch in range(epochs):
        print("EPOCH: ", epoch+1)
        set_seed(epoch)
        train_loss = train_step(model, train_dl, criterion, optimizer, scheduler, device)
        valid_loss = test_step(model, valid_dl, criterion, device)

        train_loss = train_loss / len(train_dl)
        valid_loss = valid_loss / len(valid_dl)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}')
        training_results['train_losses'].append(train_loss)
        training_results['valid_losses'].append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            # save best model
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
        # wandb.log({
        #     "Training loss": train_loss,
        #     "Validation loss": valid_loss,
        #     "Epoch": epoch+1,
        # })
        # checkpoint saver
        # checkpoint_saver(model, epoch, valid_loss)
    # training_results['path_to_weights'] = checkpoint_saver.top_model_paths[0]['path']
    return training_results
