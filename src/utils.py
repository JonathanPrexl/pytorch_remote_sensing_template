import warnings

import xarray as xr
import numpy as np
import rasterio as rio
import glob, os
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

def s2toRGB(x):
    B,G,R = x[:3]
    X = np.stack([R,G,B],axis=-1)
    X = np.clip(X*4,0,1)
    return X

def load_s1_image(path):

    # load the raw data
    with rio.open(path,"r") as src:
        d = src.read()

    if not d.shape[0] == 2:
        raise ValueError("Nope")

    polarizationnames = ["VV","VH"]
    xarr = xr.DataArray(d, dims=("polarization","lat", "long"),
                           coords={"polarization": polarizationnames})

    return xarr

def load_s2_image(path):

    # load the raw data
    with rio.open(path,"r") as src:
        d = src.read()

    if not d.shape[0] == 13:
        raise ValueError("Nope")

    bandnames = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
    xarr = xr.DataArray(d, dims=("channel","lat", "long"), coords={"channel": bandnames})

    return xarr

def load_lc(path):
 
    # load the raw data
    with rio.open(path,"r") as src:
        assert src.count == 4
        d = src.read(1)

    return d



def preprocess_s1(s1):
    
    # first restrict the values from -35 to 0
    # then add 35 so we are in the [0,35] window
    # with 0 beeing the lowest refection
    # then divide by 35 to get it into [0,1]

    return (np.clip(s1,-35,0)+35)/35   

def preprocess_s2(s2):
    
    # devide by 10k to get the 0-100% reflection
    # window in the... then clip to [0,1]

    return np.clip(s2/10000,0,1)
    # return s2/10000


def preprocess_lc(lc):
    
    # its encoded in 1-17 so we substact one

    lc = lc -1

    assert lc.min() >= 0
    assert lc.max() <= 16

    return lc



# # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md

# def weight_histograms_conv2d(writer, step, weights, layer_number, modality):
#     weights_shape = weights.shape
#     num_kernels = weights_shape[0]
#     for k in range(num_kernels):
#         flattened_weights = weights[k].flatten()
#         tag = f"{modality}_layer_{layer_number}/kernel_{k}"
#         writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

# def weight_histograms_linear(writer, step, weights, layer_number, modality):
#     flattened_weights = weights.flatten()
#     tag = f"{modality}_layer_{layer_number}"
#     writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

# def weight_histograms(writer, step, model, modality):
#     #print("Visualizing model weights...")
#     # Iterate over all model layers
#     for layer_number,(name, module) in enumerate(model.named_children()):
#         pass
#         # Compute weight histograms for appropriate layer
#         if isinstance(module, nn.Conv2d):
#             weights = module.weight
#             weight_histograms_conv2d(writer, step, weights, layer_number, modality)
#         elif isinstance(module, nn.Linear):
#             weights = module.weight
#             weight_histograms_linear(writer, step, weights, layer_number, modality)
#         elif isinstance(module, nn.Sequential):
#             for layer in module:
#                 if isinstance(layer, nn.Conv2d):
#                     weights = layer.weight
#                     weight_histograms_conv2d(writer, step, weights, layer_number, modality)
#                 elif isinstance(layer, nn.Linear):
#                     weights = layer.weight
#                     weight_histograms_linear(writer, step, weights, layer_number, modality)


# def plot_confusion_matrix(cm, class_names=None):
#     """
#     Returns a matplotlib figure containing the plotted confusion matrix.
#     Args:
#     cm (array, shape = [n, n]): a confusion matrix of integer classes
#     class_names (array, shape = [n]): String names of the integer classes
#     """

#     figure = plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion matrix")
#     plt.colorbar()

#     if class_names == None:
#         class_names = [str(i) for i in range(len(cm))]

#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)

#     # Normalize the confusion matrix.
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

#     # Use white text if squares are dark; otherwise black.
#     threshold = cm.max() / 2.

#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         color = "white" if cm[i, j] > threshold else "black"
#         plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     return figure