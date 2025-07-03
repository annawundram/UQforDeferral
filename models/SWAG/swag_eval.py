# adapted from https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/imagenet/eval_swag_imagenet.py

import argparse
import os
import random
import sys
import time
import tabulate
from sklearn.metrics import roc_auc_score
import numpy as np

import torch
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from softmax_ensemble_mcdropout_LD1.ResNet50 import ResNet_50
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.AIROGS_dataloader import AIROGS
from data.AIROGS_ood_dataloader import AIROGS_ood
from SWAG import utils, losses
from swag import SWAG

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    metavar="N",
    help="input batch size (default: 4)",
)

parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    default=None,
    metavar="CKPT",
    help="checkpoint to load (default: None)",
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=10,
    metavar="N",
    help="number of samples for SWAG (default: 30)",
)

parser.add_argument("--scale", type=float, default=1.0, help="SWAG scale")
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--use_diag_bma", action="store_true", help="sample only diag variacne for BMA"
)

parser.add_argument(
    "--seed", type=int, default=0, metavar="S", help="random seed (default: 1)"
)

parser.add_argument(
    "--save_path_swag",
    type=str,
    default=None,
    required=True,
    help="path to SWAG npz results file",
)

parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
    help="path to AIROGS file",
)

args = parser.parse_args()

eps = 1e-12

args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#test_dataset = AIROGS_ood(file_path=args.data_path, transform=None)
test_dataset = AIROGS(file_path=args.data_path, t="test", transform=None)

print(f"Test dataset length: {len(test_dataset)}")

test_dataloader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, drop_last=False
)

train_dataset = AIROGS(file_path="/AIROGS.h5", t="train", transform=None)

train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, drop_last=False
)

num_classes = 2
print("Preparing model")
swag_model = SWAG(
    ResNet_50,
    no_cov_mat=not args.cov_mat,
    max_num_models=20,
    image_channels=3,
    num_classes=num_classes,
)
swag_model.to(args.device)

criterion = losses.cross_entropy

print("Loading checkpoint %s" % args.ckpt)
checkpoint = torch.load(args.ckpt)
swag_model.load_state_dict(checkpoint["state_dict"])

print("SWAG")

swag_predictions = list()

for i in range(args.num_samples):
    swag_model.sample(args.scale, cov=args.cov_mat and (not args.use_diag_bma))

    print("SWAG Sample %d/%d. BN update" % (i + 1, args.num_samples))
    utils.bn_update(train_dataloader, swag_model, verbose=True, subset=0.1)
    print("SWAG Sample %d/%d. EVAL" % (i + 1, args.num_samples))
    res = utils.predict(test_dataloader, swag_model, verbose=True)
    predictions = res["predictions"]
    targets = res["targets"]


    nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
    print(
        "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
        % (i + 1, args.num_samples, roc_auc_score(targets, predictions[:,1]) , nll)
    )

    swag_predictions.append(predictions)

swag_predictions = np.asarray(swag_predictions)

np.savez(
    args.save_path_swag,
    AUC=roc_auc_score(targets, np.mean(swag_predictions[:, :, 1], axis=0)),
    predictions=swag_predictions,
    targets=targets,
)