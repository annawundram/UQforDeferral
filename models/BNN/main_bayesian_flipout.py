# adapted from https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/examples/main_bayesian_flipout_imagenet.py
# script for training and evaluation
'''
code adapted from PyTorch examples
'''
import argparse
import tqdm
import os
import gc
import random
import shutil
import time
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.AIROGS_dataloader import AIROGS
from data.AIROGS_ood_dataloader import AIROGS_ood
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.resnet_flipout_large as resnet
from softmax_ensemble_mcdropout_LD1.ResNet50 import ResNet_50 as det_resnet
from torchsummary import summary
from utils import util
import csv
import numpy as np
from utils.util import get_rho
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryROC
from sklearn import metrics

torchvision.set_image_backend('accimage')

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data',
                    metavar='DIR',
                    default='/AIROGS.h5',
                    help='path to dataset')
parser.add_argument('--dataset',
                    metavar='dataset',
                    default='AIROGS')                    
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=50,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--val_batch_size', default=4, type=int)
parser.add_argument('-b',
                    '--batch-size',
                    default=4,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    default=False,
                    help='use pre-trained model')
parser.add_argument('--world-size',
                    default=1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/bayesian',
                    type=str)
parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/imagenet/bayesian',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument('--num_monte_carlo',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of Monte Carlo samples')
parser.add_argument(
    '--moped',
    type=bool,
    default=False,
    help='set prior and initialize approx posterior with Empirical Bayes')
parser.add_argument('--delta',
                    type=float,
                    default=0.2,
                    help='delta value for variance scaling in MOPED')

best_pauc = 0.0
len_valset = 10145
num_classes = 2


def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)

    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
    [2] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Efficient Priors for Scalable Variational Inference in Bayesian Deep Neural Networks. ICCV workshops 2019.
    """

    if (str(layer) == 'Conv2dFlipout()'
            or str(layer) == 'Conv2dReparameterization()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data

    elif (str(layer) == 'LinearFlipout()'
          or str(layer) == 'LinearReparameterization()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors

        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_pauc
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_pauc = checkpoint['best_pauc']
            if args.gpu is not None:
                # best_pauc may be from a checkpoint from a different GPU
                best_pauc = best_pauc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    # Data loading code
    train_dataset = AIROGS(file_path=args.data, t="train", transform=None)
    valid_dataset = AIROGS(file_path=args.data, t="val", transform=None)

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
    len_trainset = len(train_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    if args.evaluate:
        validate(valid_loader, model, criterion, args)
        return

    if args.mode == 'train':

        if (args.moped):
            print("MOPED enabled")
            det_model = torch.nn.DataParallel(
                det_resnet.__dict__[args.arch](pretrained=True))
            det_model.cuda()

            for (idx_1, layer_1), (det_idx_1, det_layer_1) in zip(
                    enumerate(model.children()),
                    enumerate(det_model.children())):
                MOPED_layer(layer_1, det_layer_1, args.delta)
                for (idx_2, layer_2), (det_idx_2, det_layer_2) in zip(
                        enumerate(layer_1.children()),
                        enumerate(det_layer_1.children())):
                    MOPED_layer(layer_2, det_layer_2, args.delta)
                    for (idx_3, layer_3), (det_idx_3, det_layer_3) in zip(
                            enumerate(layer_2.children()),
                            enumerate(det_layer_2.children())):
                        MOPED_layer(layer_3, det_layer_3, args.delta)
                        for (idx_4, layer_4), (det_idx_4, det_layer_4) in zip(
                                enumerate(layer_3.children()),
                                enumerate(det_layer_3.children())):
                            MOPED_layer(layer_4, det_layer_4, args.delta)
                            for (idx_5,
                                 layer_5), (det_idx_5, det_layer_5) in zip(
                                     enumerate(layer_4.children()),
                                     enumerate(det_layer_4.children())):
                                MOPED_layer(layer_5, det_layer_5, args.delta)
                                for (idx_6,
                                     layer_6), (det_idx_6, det_layer_6) in zip(
                                         enumerate(layer_5.children()),
                                         enumerate(det_layer_5.children())):
                                    MOPED_layer(layer_6, det_layer_6,
                                                args.delta)

            del det_model

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args,
                  tb_writer)

            # evaluate on validation set
            pauc = validate(valid_loader, model, criterion, epoch, args,
                            tb_writer)

            # remember best best_pauc and save checkpoint
            is_best = pauc > best_pauc
            best_pauc = max(pauc, best_pauc)

            if is_best:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_pauc': best_pauc,
                        'optimizer': optimizer.state_dict(),
                    },
                    is_best,
                    filename=os.path.join(
                        args.save_dir,
                        'bayesian_flipout_{}.pth'.format(args.arch)))

    elif args.mode == 'test':

        if args.dataset == "AIROGS":
            path = "/AIROGS.h5"
            test_dataset = AIROGS(file_path=path, t="test", transform=None)
            print(f"Test dataset length: {len(test_dataset)}")
            test_dataloader = DataLoader(
                test_dataset, batch_size=4, shuffle=False, drop_last=False
                )
        elif args.dataset == "blur":
            path = "/ood_blur.h5"
            test_dataset = AIROGS_ood(file_path=path, transform=None)
            print(f"Test dataset length: {len(test_dataset)}")
            test_dataloader = DataLoader(
                test_dataset, batch_size=4, shuffle=False, drop_last=False
                )
        elif args.dataset == "noise":
            path = "/ood.h5"
            test_dataset = AIROGS_ood(file_path=path, transform=None)
            print(f"Test dataset length: {len(test_dataset)}")
            test_dataloader = DataLoader(
                test_dataset, batch_size=4, shuffle=False, drop_last=False
                )
        else:
            raise ValueError('Wrong dataset')

        
        checkpoint_file = '/bayesian_flipout_resnet50.pth'
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        res = evaluate(test_dataloader, model, 10)
        predictions = res["predictions"]
        targets = res["targets"]

        np.savez(
            args.save_dir,
            predictions=predictions,
            targets=targets,
        )
        


def train(train_loader, model, criterion, optimizer, epoch, args, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    global opt_th
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        '''
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        '''
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, kl = model(images)

        cross_entropy_loss = criterion(output, target)
        scaled_kl = kl.item() / args.batch_size
        elbo_loss = cross_entropy_loss + scaled_kl
        loss = cross_entropy_loss + scaled_kl

        output = output.float()
        loss = loss.float()

        # measure acc, AUC and record loss
        acc1 = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl, epoch)
            tb_writer.add_scalar('train/elbo_loss', elbo_loss.item(), epoch)
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', acc1[0].item(), epoch)
            tb_writer.flush()


def validate(val_loader, model, criterion, epoch, args, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    roc = BinaryROC()
    roc = roc.to("cpu")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    preds_list = []
    labels_list = []
    unc_list = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, kl = model(images)
            cross_entropy_loss = criterion(output, target)
            scaled_kl = (kl.item() / args.batch_size)
            elbo_loss = cross_entropy_loss + scaled_kl
            loss = cross_entropy_loss + scaled_kl

            output = output.float()
            loss = loss.float()

            # measure accuracy, AUC and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            roc.update(output[:, 1], target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        pauc = compute_pAUC(roc)
        del roc
        print(' * acc {top1.avg:.3f}'.format(top1=top1))
        print("pAUC: ", pauc)

    return pauc


def evaluate(loader, model, samples):
    predictions = list()
    targets = list()
    
    loader = tqdm.tqdm(loader)

    with torch.no_grad():
        for input, target in loader:
            predictions_samples = list()
            input = input.cuda(non_blocking=True)
            for _ in range(samples):
                output, _ = model(input)
                predictions_samples.append(F.softmax(output, dim=1).cpu().numpy())
                
            predictions_samples = np.stack(predictions_samples)  # samples x batch_size x num_classes
            predictions.append(predictions_samples) 

            del input, output, predictions_samples
            torch.cuda.empty_cache()
            gc.collect()
            
            targets.append(target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=1)  # samples x total_size x num_classes
    predictions = np.swapaxes(predictions, 0, 1)
    targets = np.concatenate(targets)
    return {"predictions": predictions, "targets": targets}


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(int(batch))]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def compute_pAUC(roc, specificity_range=(0.9, 1.0)):
        fpr, tpr, _ = roc.compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        specificity = 1 - fpr

        mask = (specificity >= specificity_range[0]) & (specificity <= specificity_range[1])
        selected_fpr = fpr[mask]
        selected_tpr = tpr[mask]

        if len(selected_fpr) > 1:
            pAUC = metrics.auc(selected_fpr, selected_tpr)
            # normalise to [0,1] by max possible AUc in this range
            pAUC = pAUC / (specificity_range[1] - specificity_range[0])
            pAUC = torch.from_numpy(np.array(pAUC))
        else:
            pAUC = torch.tensor(0.0)

        return pAUC


if __name__ == '__main__':
    main()