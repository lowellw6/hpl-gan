# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:14:27 2020

@author: Eric Bianchi
"""
from model import createDeepLabv3
from tqdm import tqdm
import torch
import numpy as np
import datahandler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix

batchsize = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load(f'./weights_16.pt', map_location=torch.device('cpu'))

model.to(device)
model.eval()   # Set model to evaluate mode

dataloaders = datahandler.get_dataloader_sep_folder(f'./', batch_size=batchsize)
nnnnn = dataloaders['Test']
n = 0
confm_sum = np.zeros((4,4))
iOU_sum = 0
f1_sum = 0

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# Iterate over data.
for batch in tqdm(iter(dataloaders['Test'])):
    
    # These lines appear to be correct.
    images = batch['image'].to(device)
    true_masks = batch['mask'].to(device, dtype=torch.long)

    # track history if only in train
    with torch.set_grad_enabled(False):
        # currently giving a tensor of [7,512,512,3]
        # where [batch, Dim, Dim, Channel]
        images = images.permute(0, 3, 1, 2).contiguous()
        # imnp = images.cpu().numpy()
        mask_pred = model(images)
        y_pred_tensor = mask_pred['out']
        
        pred = torch.argmax(y_pred_tensor, dim=1)
        y_pred = pred.data.cpu().numpy()
        # y_pred = mask_pred['out'].data.cpu().numpy()[0]
        # pred = torch.argmax(y_pred, dim=1)
        
        # This is set up perfectly for the use of determining accuracy
        y_pred = y_pred.ravel()
        y_true = true_masks.data.cpu().numpy().ravel()
        
        confm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
        iOU = jaccard_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        
    confm_sum +=confm
    iOU_sum += iOU
    f1_sum += f1
    n += 1

iOU = iOU_sum / n
f1 = f1_sum / n

plot_confusion_matrix(confm_sum, target_names=['Background', 'Steel', 'Concrete', 'Metal Deck'], normalize=True, 
                      title='Confusion Matrix')
    