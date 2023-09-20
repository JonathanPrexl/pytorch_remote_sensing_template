import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import itertools

def s2toRGB(x):
    B,G,R = x[:3]
    X = np.stack([R,G,B],axis=-1)
    X = np.clip(X*4,0,1)
    return X

def load_s2_image(path):

    # load the raw data
    with rio.open(path,"r") as src:
        d = src.read()

    if not d.shape[0] == 13:
        raise ValueError("item does not have 14 bands")

    return d

def preprocess_s2(s2):
    
    # devide by 10k to get the 0-100% reflection
    # window in the... then clip to [0,1]

    return np.clip(s2/10000,0,1)


def plot_confusion_matrix(cm, class_names=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    if class_names == None:
        class_names = [str(i) for i in range(len(cm))]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure