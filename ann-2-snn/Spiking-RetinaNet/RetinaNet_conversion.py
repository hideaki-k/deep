import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['CMU']
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# SILENCE WARNINGS
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# GPU TEST 
print(tf.__version__)
print(tf.config.list_physical_devices())
print('Built with cuda: ', tf.test.is_built_with_cuda())
print('Built with gpu support: ', tf.test.is_built_with_gpu_support())
print('gpus: ', tf.config.list_physical_devices('GPU'))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
##### PARAMETERS ###############################################
library_path = '.' #'./Github' # 
bbone_path = '.'
path_wd = './RNet_to_SNN_files' # <--- Data saving path
weights_dir = "pretrained_weights/avgpool" # <-- NN parameters path
batch_size = 1
image_shape = [896,1152]
verbose = 1
################################################################

if not os.path.exists(path_wd):
    os.mkdir(path_wd)

import site
import sys
site.addsitedir(os.path.join(library_path,'snn_toolbox'))
bbone_path = os.path.join(bbone_path, 'bbone_AVG')


from my_functions.retinanet_functions import swap_xy, convert_to_xywh, convert_to_corners, compute_iou, visualize_detections, AnchorBox, random_flip_horizontal, resize_and_pad_image, preprocess_data, LabelEncoder, DecodePredictions, RetinaNetLoss, RetinaNetLoss_Norm, RetinaNetClassificationLoss, RetinaNetBoxLoss, adapt_norm_to_output
from my_functions.retinanet_class import get_backbone, RetinaNet

label_encoder = LabelEncoder()
learning_rate_fn = 0.001 #placeholder

tar_shape = image_shape+[3]

num_classes = 80

loss_fn = RetinaNetLoss(num_classes, batch_size=batch_size)
optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

print('\n>1> Generating RetinaNet model')
resnet_backbone = get_backbone(bbone_path, input_shape = tar_shape)
model = RetinaNet(num_classes, resnet_backbone)
model.compile(loss=loss_fn, optimizer=optimizer)

print('      [ Model generated ] ')

print('\n>3> Load pretrained weights')
model.load_weights(tf.train.latest_checkpoint(weights_dir))

# BUILD MODEL

# tf.keras.backend.clear_session()
image = tf.keras.Input(shape=tar_shape, name="image")
predictions = model(image, training=False)
pred_model = tf.keras.Model(inputs=image, outputs=predictions)

model.summary()

from importlib import import_module
from snntoolbox.parsing.model_libs.keras_input_lib import load
from snntoolbox.parsing.model_libs.keras_input_lib import ModelParser
from snntoolbox.bin.utils import load_config

config_defaults_path = os.path.join(library_path,'snntoolbox/config_defaults')

print('\n>4> Create config file for SNN_Toolbox')

model_name = 'RNet'

import configparser
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'filename_ann': model_name      # Name of input model.
}

config['input'] = {
    'poisson_input': False,
    'norm_conv': False
}

config['tools'] = {
    'normalize': True               # Normalize weights for full dynamic range.
}

config['normalization'] = {
    'num_to_norm': 1,
    'percentile': 99.99,
    'method': 1,    
}

config['simulation'] = {
    'duration': 1000, #3000,                 # Number of time steps to run each sample.
    'batch_size': batch_size,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['conversion'] = {
    'spike_code': 'temporal_mean_rate',
    'max2avg_pool': True
}

config['output'] = {
    'plot_vars': {                  # Various plots (slows down simulation).
        'v_mem',
        'error_t',
        'correl',
        }
}

config['custom'] = {
    'relu_pred': False
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

config = load_config(config_defaults_path)
config.read(config_filepath)

config_filepath = os.path.join(path_wd, 'config_FULL')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)


#  3============ Parsing ===================
print('\n>6> Parse the model')

model_parser = ModelParser(model, config)
name_map = {}
idx = 0
NN = model_parser.get_layer_iterable()

# FPN -------------------------------------------------------------------------------------------------------------------------

#> Backbone:
print('\n------------------------------------\n>> FPN\n------------------------------------')
subnet = NN[0].layers[0]
bbone_out_layers = [subnet.get_layer(layer_name) for layer_name in ["conv3_block4_add", "conv4_block6_add", "conv5_block3_add", "conv5_block3_add"]]
print('ResNet output layers = ', [layer.name for layer in bbone_out_layers], '\n')
idx,bbone_tails = model_parser.parse_subnet(subnet.layers, idx, prev_out_idx=None, in_layers=None, out_layers=bbone_out_layers)
print(idx, ' - ', bbone_tails)

#> Rest:
subnet = NN[0].layers[1:]
FPN_in = [subnet[i] for i in range(3)] + [subnet[10]]
FPN_out = [subnet[i] for i in range(-6,0) if i != -2]
print('FPN input layers = ', [layer.name for layer in FPN_in])
print('FPN output layers = ', [layer.name for layer in FPN_out], '\n')
idx,FPN_tails = model_parser.parse_subnet(subnet, idx, prev_out_idx=bbone_tails, in_layers=FPN_in, out_layers = FPN_out, repair=[subnet[-1]], special_relu=[subnet[-2]])
print(idx, ' - ', FPN_tails)

# Head 1 ----------------------------------------------------------------------------------------------------------------------

print('\n------------------------------------\n>> Classification subnet:\n------------------------------------')
subnet = NN[1].layers
resh = [NN[3]]
cat = [NN[5]]
head1_in = [subnet[0]]
head1_out = [subnet[-1]]
head1_tails = []
for tail in FPN_tails:
    idx, out_ref = model_parser.parse_subnet(subnet, idx, prev_out_idx=[tail], in_layers=head1_in, out_layers = head1_out)
    idx, out_ref = model_parser.parse_subnet(resh, idx, prev_out_idx=out_ref, in_layers=resh, out_layers = resh)
    head1_tails.append(out_ref[0])
print(idx, ' - ', head1_tails)

idx, head1_tails = model_parser.parse_subnet(cat, idx, prev_out_idx=head1_tails, in_layers=cat*5, out_layers = cat)
print(idx, ' - ', head1_tails)

# Head 2 ----------------------------------------------------------------------------------------------------------------------

print('\n------------------------------------\n>> Box-regression subnet:\n------------------------------------')
subnet = NN[2].layers
resh = [NN[4]]
cat = [NN[6]]
head2_in = [subnet[0]]
head2_out = [subnet[-1]]
head2_tails = []
for tail in FPN_tails:
    idx, out_ref = model_parser.parse_subnet(subnet, idx, prev_out_idx=[tail], in_layers=head2_in, out_layers = head2_out)
    idx, out_ref = model_parser.parse_subnet(resh, idx, prev_out_idx=out_ref, in_layers=resh, out_layers = resh)
    head2_tails.append(out_ref[0])
print(idx, ' - ', head2_tails)

idx, head2_tails = model_parser.parse_subnet(cat, idx, prev_out_idx=head2_tails, in_layers=cat*5, out_layers = cat)
print(idx, ' - ', head2_tails)

# Output ----------------------------------------------------------------------------------------------------------------------

subnet = [NN[7]]
idx, NN_tail = model_parser.parse_subnet(subnet, idx, prev_out_idx=head2_tails+head1_tails, in_layers=subnet*2, out_layers = subnet)


parsed_model = model_parser.build_parsed_RNet(loss_fn, optimizer)