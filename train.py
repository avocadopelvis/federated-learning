# pip install nilearn
# pip install keras_unet_collection

# load libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import random
import os
import cv2
import glob 

# PIL adds image processing capabilities to your Python interpreter.
import PIL
from PIL import Image, ImageOps

# Shutil module offers high-level operation on a file like a copy, create, and remote operation on the file.
import shutil

# skimage is a collection of algorithms for image processing and computer vision.
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize

# NEURAL IMAGING
import nilearn as nl
import nibabel as nib # access a multitude of neuroimaging data formats
# import nilearn.plotting as nlplt
# import gif_your_nifti.core as gif2nif

# ML Libraries
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
from keras_unet_collection import losses
from keras_unet_collection import models

import warnings
warnings.filterwarnings('ignore')

# make numpy printouts easier to read
np.set_printoptions(precision = 3, suppress = True)


# dataset path
train_data = "BraTS 2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
valid_data = "BraTS 2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/"

# list of directories
train_val_directories = [f.path for f in os.scandir(train_data) if f.is_dir()]

# remove BraTS20_Training_355 since it has ill formatted name for seg.nii file
train_val_directories.remove(train_data + 'BraTS20_Training_355')

ids = pathListIntoIDs(train_val_directories)

# split ids into train+test and validation
train_test_ids, val_ids = train_test_split(ids, test_size = 0.2, random_state = 42)
# split train+test into train and test                                           
train_ids, test_ids = train_test_split(train_test_ids, test_size = 0.15, random_state = 42)



# define segmentation areas
SEGMENT_CLASSES = {
    0 : 'NOT TUMOR',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING TUMOR CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include
IMG_SIZE = 128


# create clients
clients = create_client(train_ids, 3)
valid_generator = DataGenerator(val_ids)


# HYPERPARAMETERS
loss = "categorical_crossentropy",
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
metrics = ['accuracy', losses.dice]

# add callback for training process
callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                              patience=2, min_lr=0.000001, verbose=1) 
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
    ]

# number of global epochs
rounds = 5
batch_size = 32


# U-Net
# model = models.unet_2d((IMG_SIZE, IMG_SIZE, 2), [32, 64, 128, 256, 512], 4)
# U-Net++
# model = models.unet_plus_2d((IMG_SIZE, IMG_SIZE, 2), [32, 64, 128, 256, 512], 4)
# Attention U-Net
model = models.att_unet_2d((IMG_SIZE, IMG_SIZE, 2), [32, 64, 128, 256, 512], 4)


# TRAINING

# initialize global model
# global_model = build_unet(input_layer, 'he_normal', 0.2)
global_model = model
global_model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
        )
print("Begin Training")
# commence global training loop
for round in range(1, rounds+1):
  print(f'\nRound: {round}')

  # get global model's weights
  global_weights = global_model.get_weights()

  # initial list to collect local model weights after scaling
  scaled_local_weight_list = list()

  # get client names
  client_names= list(clients.keys())
  random.shuffle(client_names)

  count = 1
  # loop through each client and create new local model
  for client in client_names:
    print(f'Client {count}')
    local_model = model
    local_model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
        )
    
    #set local model weight to the weight of the global model
    local_model.set_weights(global_weights)

    # get client data and pass it through a data generator
    data = DataGenerator(clients[client])

    # fit local model with client's data
    local_model.fit(data, epochs = 1, steps_per_epoch = len(data), verbose = 1) #callbacks = callbacks, validation_data = valid_generator)

    # scale the model weights and add to list
    scaling_factor = weight_scaling_factor(data)
    scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
    
    # not adding scaling
    scaled_local_weight_list.append(local_model.get_weights())

    # clear session to free memory after each communication round
    K.clear_session()

    count += 1

  #to get the average over all the local model, we simply take the sum of the scaled weights
  average_weights = sum_scaled_weights(scaled_local_weight_list)
      
  #update global model 
  global_model.set_weights(average_weights)

  evaluate_model(test_ids, global_model, round)

print('\nTraining Done!')


















