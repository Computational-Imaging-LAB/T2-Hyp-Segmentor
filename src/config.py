# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:51:55 2020

@author: trabz
"""

import numpy as np




# These variables are the hyperparameters of training

## Batch size of training.
BATCH=1

## Input images will be resized to [256*256*d]
IMAGE_SIZE=256

##This is the input channels of network. Each modality added as a new channel
INPUT_CHANNELS=4

## It is output classes. Each an every of channels indicates classes.
OUTPUT_CHANNELS=7

## Image rotation for augmentation. This value determines the angle of the rotation
AUG_ROTATE=None

## Image flip for augmentation. This value determines the probability. Must be between 0-1
AUG_FLIP=None

## Preprocess of data. normalize,crop,rescale,pad can be selected. For details look to the documentation
PREPROCESS=None

## Epochs of training
EPOCHS=20

## Data sampling batch. random and uniform can be selected
DATA_SAMPLING='random'

# These variables are the dataset parameters
DATAPATH=r"F:\181273_407317_bundle_archive\training\*\*"

## Input modalities or channels of the data. For one input insert one path
MODALITY_PATHS=['\\*flair*.nii.gz','\\*1_t1.nii.gz','\\*_1_t1ce.nii.gz','\\*_1_t2.nii.gz']

## Ground truth of segmented images
MASK_PATH='\\segmentedMask.nii'

## Validation datapath of the data
VAL_DATAPATH=r"F:\181273_407317_bundle_archive\val\*\*"

## Test datapath of the data
TEST_DATAPATH=r"F:\181273_407317_bundle_archive\val\*\*"

## learning rate of model
LR = 0.0001

## workers num. 0 for windows.
WORKERS = 0

## Output folder for saved model
WEIGHTS = "/"

###
# Determine the dimensions of input data 3 for [w,h,d] nad 2 for [w,h] 
DIMENSION_TYPE=3 