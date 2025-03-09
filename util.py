import torch
import time
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import math
from collections import OrderedDict
import json

loaders = dict()
sets = dict()

def get_loaders():
    return loaders

def get_sets():
    return sets

def build_data_loaders(args):
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
    
def get_model(args):
    mod_name = args.model[0]
    ctor = getattr(models, mod_name)
    model = ctor(pretrained=True)
    model.arch = mod_name

    for param in model.parameters(): #freeze parms
        param.requires_grad = False
    
    print(model)
    return model

def load_chkpt(args, model):
    if torch.cuda.device_count():
        checkpoint = torch.load(args.chkpt_pth)
    else: #map_location=torch.device("cpu"
        checkpoint = torch.load(args.chkpt_pth, map_location=torch.device("cpu"))
        
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

def do_test(args, model):
    test_loader = loaders['test']
    print("testing")
    correct = 0
    total = 0
    n = 0
    with torch.no_grad():
        for data in test_loader:
            print(f"\t n = {n}");n += 1
            images, labels = data[0].to(args.dev), data[1].to(args.dev)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    img_cnt = len(sets['test'])
    print(f"Accuracy of the network on the {img_cnt} test images {accuracy}%")

def common_train_predict_args(parser):
    parser.add_argument('--data-dir', action="store", default='',
                        help = "where the flowers jps are(no trailing /)")
    parser.add_argument('--dev', nargs=1, choices=['cpu', 'cuda'], default='cpu',
                        help = "run model where?")
    parser.add_argument('--model', nargs=1, choices=['vgg16'], default='vgg16',
                        help="model name to import from torch")
    parser.add_argument('--chkpt-pth', action="store",  default='/tmp/chkpt.pth',
                        help="the path of the checkpoint file")

def get_cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
