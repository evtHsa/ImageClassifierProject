


> Written with [StackEdit](https://stackedit.io/)
## Intro
This is the final project in the Udacity AI Programming with Python nanodegree program.

The purpose is to use transfer learning to adapt a pretrained model with "image recognition capabilities" (example vgg, alexnet) by replacing the last layer with a classifier that we then train on the provided flowers data set
## Required Submission Files
### train.py

 - **Train** the classifier layer added to the pretrained net starting from either
	 - Just the pretrained network
	 - The checkpoint file saved periodically
#### Usage
usage: train.py [-h] [--data-dir DATA_DIR] [--dev {cpu,cuda}] [--model {vgg16}]
                [--chkpt-pth CHKPT_PTH] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
                [--criterion {NLLLoss}] [--start-from-chkpt] [--optimizer {Adam}]
                [--chkpt-every CHKPT_EVERY] [--print-every PRINT_EVERY]

Example with long option names

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   where the flowers jps are(no trailing /)
  --dev {cpu,cuda}      run model where?
  --model {vgg16}       model name to import from torch
  --chkpt-pth CHKPT_PTH
                        the path of the checkpoint file
  --epochs EPOCHS       1 epoch is 1 pass through the training data
  --batch-size BATCH_SIZE
                        1 step through the training data
  --lr LR               learning rate
  --criterion {NLLLoss}
                        the loss function
  --start-from-chkpt    start from a saved checkpoint
  --optimizer {Adam}    Used to adjust weights
  --chkpt-every CHKPT_EVERY
                        checckpoint every this many training steps
  --print-every PRINT_EVERY
                        print stats every this many training steps

### predict.py
Take a path to an image and a model with a trained classifier and display the image and the K top probabilities for classification
#### Usage
usage: predict.py [-h] [--data-dir DATA_DIR] [--dev {cpu,cuda}] [--model {vgg16}]
                  [--chkpt-pth CHKPT_PTH] [--img-path IMG_PATH] [--num-random-imgs NUM_RANDOM_IMGS]
                  [--top-k TOP_K]

Example with long option names

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   where the flowers jps are(no trailing /)
  --dev {cpu,cuda}      run model where?
  --model {vgg16}       model name to import from torch
  --chkpt-pth CHKPT_PTH
                        the path of the checkpoint file
  --img-path IMG_PATH   path to jpg file to classify(if not using num-ramdom-imgs
  --num-random-imgs NUM_RANDOM_IMGS
                        how many images to classify
  --top-k TOP_K         how many probabilities to show

### util.py
Various utility functions used by train or predict or both.

## Additional Submission files
### Image Classifier Project.ipynb
The jupyter notedbook from part 1 of the project where the initial versions of what is seen in <train, predict, util>.py were developed
### Training log
Documenting the training from about 50% accuracy, when I ran very low on gpu time, to nearly 90% with about 6 days compute time on an old Intel Core i7 cpu.

## Experiences with Training
Initially I had to large a learning rate (0.3) and accuracy oscillated and I used far to much GPU time before eventually cutting it to 1.0e-5(0.00001).
