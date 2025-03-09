#python predict.py --data-dir ~/tmp/flowers/test --dev cpu --chkpt-pth /home/evt/wrk/udacity/trash/saved_chkpt.pth --img-path 21/image_06805.jpg --top-k 3
#
# predict.py --data-dir ~/tmp/flowers/test --dev cpu --chkpt-pth /home/evt/wrk/udacity/trash/saved_chkpt.pth --img-path test/21/image_06805.jpg --num-random-imgs 3
#
# python predict.py --data-dir ~/tmp/flowers/test --dev cpu --chkpt-pth /home/evt/wrk/udacity/trash/saved_chkpt.pth --img-path 21/image_06805.jpg
#
import torch
from PIL import Image
import seaborn as sb
import matplotlib.pyplot as plt
import argparse
import util as ut
import numpy as np
import os
import random

def get_scaled_image(image_path):
    im = Image.open(image_path)
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
    dims = (list(im.size)) #[width, height] 
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail
    short_side = dims.index(min(dims))
    dims[short_side] = 256 # shortest side now 256
    MAXINT_32 = 2**31 - 1
    if short_side == 0: # width
        im.thumbnail((dims[short_side],MAXINT_32), Image.ANTIALIAS)
    else:
        im.thumbnail((MAXINT_32, dims[short_side]), Image.ANTIALIAS)
        
    w, h = dims = im.size #im resize in place, get new dims, tuple because dont need to alter
    model_img_size = 224

    top = (h - model_img_size)/2
    left = (w - model_img_size)/2
    im = im.crop((left, top, left + model_img_size, top + model_img_size))
    return im               

def gen_np_img(pil_image, mean, stdev):
    np_image = np.array(pil_image) / 255
    # per instructions above You'll want to subtract the means from each color channel, then divide by the 
    # standard deviation.
    np_image -= np.array(mean) 
    np_image /= np.array(stdev)
    np_image= np_image.transpose((2,0,1))
    return np_image
                          
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = get_scaled_image(image_path)
    np_im = gen_np_img(im, [0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    return np_im


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image, (1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax

def predict(image_path, model, topkl):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path) #loading image and processing it using above defined function
    im = torch.from_numpy(image).type(torch.FloatTensor) # np array -> tensor
    im = im.unsqueeze(dim = 0) # batch size == 1 for inference 
        
    with torch.no_grad():
        output = model.forward(im)
        
    logps = torch.exp(output)
    probs, indices = logps.topk(topkl)
    probs = probs.numpy() # tensor -> nparray
    indexes = indices.numpy() 
    
    prob_list = probs.tolist()[0]
    ix_list = indices.tolist()[0]
    mapping = {val: key for key, val in model.class_to_idx.items()}
    
    classes = [mapping [item] for item in ix_list]
    classes = np.array(classes)
    
    return probs, classes

def get_random_images(args):

    # get a list of all file names
    hit_list = []

    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                hit_list.append(os.path.join(root, file))
    # pick N at random and return in a list
    return random.sample(hit_list, int(args.num_random_imgs))

def test(args, model):
    path_list = list()
    if not args.num_random_imgs:
        path_list.append(args.data_dir + '/' + args.img_path)
    else:
        path_list = get_random_images(args)

    import pdb;pdb.set_trace
    for img_path in path_list:
        image = process_image(img_path)
        imshow(image)
    
        probs, classes = predict(img_path, model, int(args.top_k))
        cat_to_name = ut.get_cat_to_name()
        class_names = [cat_to_name [item] for item in classes]
        plt.figure(figsize = (6,10))
        plt.subplot(2,1,2)
        probs = np.reshape(probs, -1)

        for i in range(len(probs)):
            print(f"{class_names[i]}: probability: {probs[i]:e}")

        sb.barplot(x=probs, y=class_names, color= 'red');
        plt.show()

def main(args):
    
    model = ut.get_model(args)
    ut.load_chkpt(args, model)
    model.to(args.dev[0]);    
    test(args, model)

#############################################################################
parser = argparse.ArgumentParser(description='Example with long option names')
ut.common_train_predict_args(parser)
parser.add_argument('--img-path', action="store", default='',
                    help='path to jpg file to classify(if not using num-ramdom-imgs')
parser.add_argument('--num-random-imgs', action="store", default=1,
                    help="how many images to classify")

parser.add_argument('--top-k', action="store", default=3,
                    help="how many probabilities to show")

args = parser.parse_args()

import pdb;pdb.set_trace
main(args)
