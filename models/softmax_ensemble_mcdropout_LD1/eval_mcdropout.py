from torch.utils.data import DataLoader
from AIROGS_dataloader import AIROGS
from AIROGS_ood_dataloader import AIROGS_ood
import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import pandas as pd
import gc
import argparse
import tqdm
from McDropout import McDropoutModel


# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(loader, model, samples, verbose=False):
    predictions = list()
    targets = list()
    
    if verbose:
        loader = tqdm.tqdm(loader)

    with torch.no_grad():
        for input, target in loader:
            predictions_samples = list()
            input = input.cuda(non_blocking=True)
            predictions_samples = model.sample(input, samples)
            
            predictions.append(predictions_samples.cpu().numpy()) 

            del input, predictions_samples
            torch.cuda.empty_cache()
            gc.collect()
            
            targets.append(target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=1)  # samples x total_size x num_classes
    predictions = np.swapaxes(predictions, 0, 1)
    targets = np.concatenate(targets)
    return {"predictions": predictions, "targets": targets}


def run(dataset, save_path, save_name):

    if dataset == "AIROGS":
        path = "/AIROGS.h5"
        test_dataset = AIROGS(file_path=path, t="test", transform=None)
        print(f"Test dataset length: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset, batch_size=4, shuffle=False, drop_last=False
            )
    elif dataset == "blur":
        path = "/ood_blur.h5"
        test_dataset = AIROGS_ood(file_path=path, transform=None)
        print(f"Test dataset length: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset, batch_size=4, shuffle=False, drop_last=False
            )
    elif dataset == "noise":
        path = "/ood.h5"
        test_dataset = AIROGS_ood(file_path=path, transform=None)
        print(f"Test dataset length: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset, batch_size=4, shuffle=False, drop_last=False
            )
    else:
        raise ValueError('Wrong dataset')

    model = McDropoutModel.load_from_checkpoint("") # load mc dropout model
    model.eval()
    model.to(device)
    
    res = predict(test_dataloader, model, 10, verbose=True)
    predictions = res["predictions"]
    targets = res["targets"]

    np.savez(
        save_path + save_name,
        predictions=predictions,
        targets=targets,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--dataset', default='AIROGS', type=str, help='dataset = [AIROGS, blur, noise, Chaksu]')
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--save_name', default='mcdropout_out', type=str)

    args = parser.parse_args()

    run(args.dataset, args.save_path, args.save_name)