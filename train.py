#

# python train.py --data-dir ~/tmp/flowers --epochs 0 --batch-size 16 --lr 0.003 --criterion NLLLoss --dev cpu --model vgg16  --chkpt-pth ~/wrk/udacity/trash/checkpoint.pth
#
import torch
import time
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import os
import math
import numpy as np
from PIL import Image
import argparse
import seaborn as sb
import json

def gt_0(val):
    v = int(val)
    if v > 0: return v
    raise argparse.ArgumentTypeError(f"{val} is not a positive integer")

def gte_0(val):
    v = int(val)
    if v >= 0: return v
    raise argparse.ArgumentTypeError(f"{val} is not greater than or equal 0")

def range_0_1(val):
    f = float(val)
    if f >= 0 and f <= 1: return f
    raise argparse.ArgumentTypeError(f"{val} is not in [0,1]")
    
def get_fn_obj(val):
    if len(val) == 1 and val[0] == 'NLLLoss':
        return nn.NLLLoss
    raise argparse.ArgumentTypeError(f"{val} is not a criterion we understand")

# 
def build_data_loaders(args, sets, loaders):
    # simplified from part 1
    train_xform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    valid_xform =transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    test_xform = valid_xform

    # create data sets
    sets['train'] = datasets.ImageFolder(args.data_dir + "/train",
                                         transform=train_xform)
    sets['test'] = datasets.ImageFolder(args.data_dir + "/test",
                                         transform=train_xform)
    sets['valid'] = datasets.ImageFolder(args.data_dir + "/valid",
                                         transform=train_xform)

    loaders['train'] = torch.utils.data.DataLoader(sets['train'],
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    # don't shuffle for test and validate
    loaders['test'] = torch.utils.data.DataLoader(sets['test'],
                                                  batch_size=args.batch_size)

    loaders['valid'] = torch.utils.data.DataLoader(sets['valid'],
                                                 batch_size=args.batch_size)

def show_loader_info(args, sets, loaders):
    for key in sets.keys():
        print(f"dataset len({key}) = " + str(len(sets[key])))
        print(f"dataldr len({key}) = " + str(len(loaders[key])))

    print(f"steps/epoch = {math.ceil(len(loaders['train']) / args.batch_size)}")

def get_classifier():
    dict = OrderedDict([('fc1', nn.Linear(25088, 4096)),
                        ('relu', nn.ReLU()), 
                        ('fc2', nn.Linear(4096, 102)),
                        ('output', nn.LogSoftmax(dim=1))])
    classifier = nn.Sequential(dict)
    return classifier
    
def get_model(args, sets):
    mod_name = args.model[0]
    ctor = getattr(models, mod_name)
    model = ctor(pretrained=True)
    model.arch = mod_name
    model.class_to_idx = sets['train'].class_to_idx
    for param in model.parameters(): #freeze parms
        param.requires_grad = False
    
    model.classifier = nn.Sequential(get_classifier())
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    model.to(args.dev);    
    print(model)
    return model

def load_chkpt(args, model):
    path = args.chkpt_pth
    if torch.cuda.device_count():
        checkpoint = torch.load(path)
    else: #map_location=torch.device("cpu"
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        
    import pdb;pdb.set_trace()
    assert(checkpoint['arch'] == args.model[0])
    
        
    model.arch         = checkpoint['arch'] 
    model.epoch        = checkpoint['epoch'] 
    model.step         = checkpoint['step'] 
    model.features     = checkpoint ['features']
    model.class_to_idx = checkpoint ['class_to_idx']
    model.classifier   = checkpoint ['classifier']
    model.optimizer    = checkpoint ['optimizer']
    model.losslog      = checkpoint['losslog']
    model.load_state_dict(checkpoint ['state_dict'])
    # no need to refreeze pretrained parameters
    
    return None # in case the unwary expect this to create and return the model


def main(args):
    print("boy howdy2")
    sets = dict()
    ldrs = dict()
    build_data_loaders(args, sets, ldrs)
    show_loader_info(args, sets, ldrs)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = get_model(args, sets)
    if args.chkpt_pth:
        load_chkpt(args, model)
    import pdb;pdb.set_trace()


#############################################################################
parser = argparse.ArgumentParser(description='Example with long option names')
parser.add_argument('--data-dir', action="store", default='')
parser.add_argument('--epochs', type=gte_0, action="store", default=3)
parser.add_argument('--batch-size', type=gt_0, action="store", default=16)
parser.add_argument('--lr', type=range_0_1, action="store", default=16)
parser.add_argument('--criterion', nargs=1, choices=['NLLLoss'], default='NLLLoss')
parser.add_argument('--dev', nargs=1, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--model', nargs=1, choices=['vgg16'], default='vgg16')
parser.add_argument('--chkpt-pth', action="store",  default='/tmp/chkpt.pth')

args = parser.parse_args()
args.criterion = get_fn_obj(args.criterion)
args.dev = torch.device(args.dev[0])

main(args)
