#

#python train.py --data-dir ~/tmp/flowers --epochs 7 --batch-size 16 --lr 0.003 --criterion NLLLoss --dev cpu --model vgg16 --chkpt-pth ~/wrk/udacity/trash/checkpoint.pth --print-every 1 --optimizer Adam
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
    model.criterion = args.criterion
    model.to(args.dev);    
    print(model)
    return model

def save_chkpt(model, epoch, step):
    import pdb;pdb.set_trace()
    checkpoint = {'arch'        : model.arch,
                  'epoch'       : epoch,
                  'step'        : step,
                  'features'    : model.features,
                  'class_to_idx': model.class_to_idx,
                  'classifier'  : model.classifier,
                  'state_dict'  : model.state_dict(),
                  'losslog'     : model.losslog
                 }
    torch.save(checkpoint, chkpt_pth)

def load_chkpt(args, model):
    path = args.chkpt_pth
    if torch.cuda.device_count():
        checkpoint = torch.load(path)
    else: #map_location=torch.device("cpu"
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        
    assert(checkpoint['arch'] == args.model[0])
    
        
    model.arch         = checkpoint['arch'] 
    model.epoch        = checkpoint['epoch'] 
    model.step         = checkpoint['step'] 
    model.features     = checkpoint ['features']
    model.class_to_idx = checkpoint ['class_to_idx']
    model.classifier   = checkpoint ['classifier']
    model.losslog      = checkpoint['losslog']
    model.load_state_dict(checkpoint ['state_dict'])
    # no need to refreeze pretrained parameters
    print("checkpoint loaded")
    return None # in case the unwary expect this to create and return the model

def show_elapsed_mins(t0):
    elapsed_min = (time.time() - t0) / 60
    print(f"\t elapsed(min) = {elapsed_min:.2f}")

def do_train(args, loaders, model):
    steps = 0
    running_loss = 0
    chkpt_every = 10
    t0 = time.time()
    train_ldr = loaders['train']
    test_ldr = loaders['test']
    valid_ldr = loaders['valid']
    x = args.optimizer[0]
    optimizer_ctor = getattr(torch.optim, args.optimizer[0])
    optimizer = optimizer_ctor(model.classifier.parameters(), lr=args.lr)

    import pdb;pdb.set_trace()
    for epoch in range(model.epoch, args.epochs): 
        print("epoch: " + str(epoch))

        for inputs, labels in train_ldr:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(args.dev), labels.to(args.dev)        
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = args.criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            if steps % args.chkpt_every == 0:
                show_elapsed_mins(t0)
                print(f"\t saving checkpt at step {steps}")
                save_chkpt(model, optimizer, epoch, steps)            

            running_loss += loss.item()

            if steps % args.print_every == 0:
                print(f"step {steps}")
                show_elapsed_mins(t0)
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_ldr:
                        inputs, labels = inputs.to(args.dev), labels.to(args.dev)
                        logps = model.forward(inputs)
                        batch_loss = args.criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
             
                stats = {'epoch' : f"{epoch+1}/{args.epochs}",
                         'train_loss'    : f"{running_loss/args.print_every:.3f}",
                         'test_loss'     : f"{test_loss/len(test_ldr):.3f} ",
                         'test_accuracy' : f"{accuracy/len(test_ldr):.3f}"}

                print(stats)
                model.losslog.append(stats)
                running_loss = 0
                model.train()
                
def main(args):
    print("boy howdy2")
    sets = dict()
    ldrs = dict()
    build_data_loaders(args, sets, ldrs)
    show_loader_info(args, sets, ldrs)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    args.criterion = getattr(nn, args.criterion[0])() # the trailing () matters
    args.dev = torch.device(args.dev[0])
    model = get_model(args, sets)
    
    if args.chkpt_pth:
        load_chkpt(args, model)
    do_train(args, ldrs, model)


#############################################################################
parser = argparse.ArgumentParser(description='Example with long option names')
parser.add_argument('--data-dir', action="store", default='')
parser.add_argument('--epochs', type=gte_0, action="store", default=3)
parser.add_argument('--batch-size', type=gt_0, action="store", default=16)
parser.add_argument('--lr', type=range_0_1, action="store", default=16)
parser.add_argument('--criterion', nargs=1, choices=['NLLLoss'], default='NLLLoss')
parser.add_argument('--optimizer', nargs=1, choices=['Adam'], default='Adam')
parser.add_argument('--dev', nargs=1, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--model', nargs=1, choices=['vgg16'], default='vgg16')
parser.add_argument('--chkpt-pth', action="store",  default='/tmp/chkpt.pth')
parser.add_argument('--chkpt-every', action="store",  default=25,
                    help="checckpoint every this many training steps")
parser.add_argument('--print-every', action="store",  default=25,
                    help="print stats every this many training steps")

args = parser.parse_args()

main(args)
