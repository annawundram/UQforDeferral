# adapted from https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/imagenet/run_swag_imagenet.py
import argparse
import os
import random
import sys
import time
import tabulate
import subprocess
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from softmax_ensemble_mcdropout_LD1.ResNet50 import ResNet_50
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.AIROGS_dataloader import AIROGS
from SWAG import utils, losses
from swag import SWAG



parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    metavar="N",
    help="input batch size (default: 4)",
)
"""
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
"""
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="pretrained model usage flag (default: off)",
)
parser.add_argument(
    "--parallel", action="store_true", help="data parallel model switch (default: off)"
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--save_freq", type=int, default=1, metavar="N", help="save frequency (default: 1)"
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=1,
    metavar="N",
    help="evaluation frequency (default: 1)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_cpu", action="store_true", help="store swag on cpu (default: off)"
)
parser.add_argument(
    "--swa_start",
    type=float,
    default=4,
    metavar="N",
    help="SWA start epoch number (default: 4)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_freq",
    type=int,
    default=4,
    metavar="N",
    help="SWA model collection frequency/ num samples per epoch (default: 4)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="CE",
    help="loss to use for training model (default: Cross-entropy)",
)

parser.add_argument(
    "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
)
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

args = parser.parse_args()


args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Loading AIROGS from %s" % (args.data_path))
train_dataset = AIROGS(file_path=args.data_path, t="train", transform=None)
valid_dataset = AIROGS(file_path=args.data_path, t="val", transform=None)

print(f"Training dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(valid_dataset)}")

def get_oversampler(dataset):
    # oversample minority class (RG)
    class_counts = torch.bincount(torch.tensor(dataset.labels))  
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, drop_last=False, sampler=get_oversampler(train_dataset)
    )
valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, drop_last=False, sampler=get_oversampler(valid_dataset)
)
num_classes = 2

print("Preparing model")
model = ResNet_50(3, num_classes)
model.to(args.device)

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True
if args.swa:
    print("SWAG training")
    args.swa_device = "cpu" if args.swa_cpu else args.device
    swag_model = SWAG(
        ResNet_50,
        no_cov_mat=args.no_cov_mat,
        max_num_models=20,
        image_channels=3,
        num_classes=num_classes,
)
    swag_model.to(args.swa_device)
    if args.pretrained:
        model.to(args.swa_device)
        swag_model.collect_model(model)
        model.to(args.device)
else:
    print("SGD training")


def schedule(epoch):
    if args.swa and epoch >= args.swa_start:
        return args.swa_lr
    else:
        return args.lr_init * (0.1 ** (epoch // 30))


# use a slightly modified loss function that allows input of model
if args.loss == "CE":
    criterion = losses.cross_entropy
    # criterion = F.cross_entropy
elif args.loss == "adv_CE":
    criterion = losses.adversarial_cross_entropy

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd
)

if args.parallel:
    print("Using Data Parallel model")
    model = torch.nn.parallel.DataParallel(model)

start_epoch = 0
if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model.load_state_dict(checkpoint["state_dict"])

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "pAUC": None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

num_iterates = 0
best_val_swa = 0.0
best_val_sgd = 0.0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    print("EPOCH %d. TRAIN" % (epoch + 1))
    if args.swa and (epoch + 1) > args.swa_start:
        subset = 1.0 / args.swa_freq
        for i in range(args.swa_freq):
            print("PART %d/%d" % (i + 1, args.swa_freq))
            train_res = utils.train_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                subset=subset,
                verbose=True,
            )

            num_iterates += 1
            utils.save_checkpoint(
                args.dir, num_iterates, name="iter", state_dict=model.state_dict()
            )

            model.to(args.swa_device)
            swag_model.collect_model(model)
            model.to(args.device)
    else:
        train_res = utils.train_epoch(
            train_loader, model, criterion, optimizer, verbose=True
        )

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        print("EPOCH %d. VAL" % (epoch + 1))
        test_res = utils.eval(valid_loader, model, criterion, verbose=True)
    else:
        test_res = {"loss": None, "pAUC": None}

    if args.swa and (epoch + 1) > args.swa_start:
        if (
            epoch == args.swa_start
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
        ):
            swag_res = {"loss": None, "pAUC": None}
            swag_model.to(args.device)
            swag_model.sample(0.0)
            print("EPOCH %d. SWAG BN" % (epoch + 1))
            utils.bn_update(train_loader, swag_model, verbose=True, subset=0.1)
            print("EPOCH %d. SWAG EVAL" % (epoch + 1))
            swag_res = utils.eval(valid_loader, swag_model, criterion, verbose=True)
            swag_model.to(args.swa_device)
        else:
            swag_res = {"loss": None, "pAUC": None}

    if (epoch + 1) % args.save_freq == 0:
        if args.swa:
            if swag_res["pAUC"] is not None and swag_res["pAUC"] > best_val_swa:
                best_val_swa = swag_res["pAUC"]
                utils.save_checkpoint(
                    args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
                )
        else:
            if test_res["pAUC"] is not None and test_res["pAUC"] > best_val_sgd:
                best_val_sgd = test_res["pAUC"]
                utils.save_checkpoint(
                    args.dir,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)
    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["pAUC"],
        test_res["loss"],
        test_res["pAUC"],
        time_ep,
        memory_usage,
    ]
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["pAUC"]] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    table = table.split("\n")
    table = "\n".join([table[1]] + table)
    print(table)

if args.epochs % args.save_freq != 0:
    if args.swa:
        if swag_res["pAUC"] is not None and swag_res["pAUC"] > best_val_swa:
            best_val_swa = swag_res["pAUC"]
            utils.save_checkpoint(
                args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
            )
    else:
        if test_res["pAUC"] is not None and test_res["pAUC"] > best_val_sgd:
            best_val_sgd = test_res["pAUC"]
            utils.save_checkpoint(
                args.dir,
                args.epochs,
                state_dict=model.state_dict(),
            )