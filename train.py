#python train.py --data-dir ~/tmp/flowers --epochs 500 --batch-size 16 --lr 0.00005 --criterion NLLLoss --dev cpu --model vgg16 --chkpt-pth ~/wrk/udacity/trash/checkpoint.pth --print-every 50 --optimizer Adam --chkpt-every 100 --start-from-chkpt
#
# try much lower learning rate
# python train.py --data-dir ~/tmp/flowers --epochs 100 --batch-size 16 --lr 0.00005 --criterion NLLLoss --dev cpu --model vgg16 --chkpt-pth ~/wrk/udacity/trash/checkpoint.pth --print-every 50 --optimizer Adam --chkpt-every 100 --start-from-chkpt
#
# python train.py --data-dir ~/tmp/flowers --epochs 40 --batch-size 16 --lr 0.003 --criterion NLLLoss --dev cpu --model vgg16 --chkpt-pth ~/wrk/udacity/trash/checkpoint.pth --print-every 50 --optimizer Adam --chkpt-every 100 --start-from-chkpt
#
from collections import OrderedDict
import os
import argparse
import util as ut
import torch
from torch import nn
import time

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
    
def save_chkpt(model, epoch, step):
    checkpoint = {'arch'        : model.arch,
                  'epoch'       : epoch,
                  'step'        : step,
                  'features'    : model.features,
                  'class_to_idx': model.class_to_idx,
                  'classifier'  : model.classifier,
                  'state_dict'  : model.state_dict(),
                  'losslog'     : model.losslog
                 }
    torch.save(checkpoint, args.chkpt_pth)

def do_train(args, loaders, model):
    if hasattr(model, 'step'):
        steps = model.step
    else:
        steps = 0
    if hasattr(model, 'epoch'):
        start_epoch = model.epoch
    else:
        start_epoch = 0
        
    running_loss = 0
    chkpt_every = 10
    t0 = time.time()
    train_ldr = loaders['train']
    test_ldr = loaders['test']
    valid_ldr = loaders['valid']
    x = args.optimizer[0]
    optimizer_ctor = getattr(torch.optim, args.optimizer[0])
    optimizer = optimizer_ctor(model.classifier.parameters(), lr=args.lr)

    for epoch in range(start_epoch, args.epochs): 
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

            if steps % int(args.chkpt_every) == 0:
                ut.show_elapsed_mins(t0)
                print(f"\t saving checkpt at step {steps}")
                save_chkpt(model, epoch, steps)            

            running_loss += loss.item()

            if steps % int(args.print_every) == 0:
                print(f"step {steps}")
                ut.show_elapsed_mins(t0)
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
             
                stats = {'epoch' : f"{epoch+1}/{int(args.epochs)}",
                         'train_loss'    : f"{running_loss/int(args.print_every):.3f}",
                         'test_loss'     : f"{test_loss/len(test_ldr):.3f} ",
                         'test_accuracy' : f"{accuracy/len(test_ldr):.3f}"}

                print(stats)
                if not hasattr(model,'losslog'):
                    model.losslog = list()
                model.losslog.append(stats)
                running_loss = 0
                model.train()
                
def main(args):
    ut.build_data_loaders(args)
    
    sets = ut.get_sets()
    ldrs = ut.get_loaders()
    
    ut.show_loader_info(args, sets, ldrs)
    
    args.criterion = getattr(nn, args.criterion[0])() # the trailing () matters
    args.dev = torch.device(args.dev[0])
    model = ut.get_model(args)
    model.class_to_idx = sets['train'].class_to_idx #all sets have it, pick this one
    model.classifier = nn.Sequential(ut.get_classifier())
    model.criterion = args.criterion

    if args.start_from_chkpt:
        ut.load_chkpt(args, model)

    model.to(args.dev);    
    do_train(args, ldrs, model)


#############################################################################
parser = argparse.ArgumentParser(description='Example with long option names')
ut.common_train_predict_args(parser)

parser.add_argument('--epochs', type=gte_0, action="store", default=3,
                    help="1 epoch is 1 pass through the training data")
parser.add_argument('--batch-size', type=gt_0, action="store", default=16,
                    help="1 step through the training data")
parser.add_argument('--lr', type=range_0_1, action="store", default=16,
                    help="learning rate")
parser.add_argument('--criterion', nargs=1, choices=['NLLLoss'], default='NLLLoss',
                    help="the loss function")
parser.add_argument('--start-from-chkpt', action='store_true',
                    help="start from a saved checkpoint")
parser.add_argument('--optimizer', nargs=1, choices=['Adam'], default='Adam',
                    help="Used to adjust weights")
parser.add_argument('--chkpt-every', action="store",  default=25,
                    help="checckpoint every this many training steps")
parser.add_argument('--print-every', action="store",  default=25,
                    help="print stats every this many training steps")

args = parser.parse_args()

main(args)
