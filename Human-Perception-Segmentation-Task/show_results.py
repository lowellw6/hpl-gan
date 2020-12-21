# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 01:01:40 2020

@author: Eric Bianchi
"""

import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd



def generate_images(model, image_path, name, destination_mask, destination_combined):
    
    if not os.path.exists(destination_mask): # if it doesn't exist already
        os.makedirs(destination_mask)
    
    if not os.path.exists(destination_combined): # if it doesn't exist already
        os.makedirs(destination_combined)  
    
    ino = 2
 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = image
    mask = img.reshape(3,512,512)
    '''
    with torch.no_grad():
        mask_pred = model(torch.from_numpy(img).type(torch.FloatTensor))
        
    # color mapping corresponding to classes
    # ---------------------------------------------------------------------
    # 0 = background
    # 1 = fair
    # 2 = poor
    # 3 = severe
    # ---------------------------------------------------------------------
    import numpy as np
    mapping = {0:np.array([0,0,0], dtype=np.uint8), 1:np.array([128,0,0], dtype=np.uint8),
               2:np.array([0,128,0], dtype=np.uint8), 3:np.array([128,128,0], dtype=np.uint8)}
    
    y_pred_tensor = mask_pred['out']
    pred = torch.argmax(y_pred_tensor, dim=1)
    y_pred = pred.data.cpu().numpy()
    
    import numpy as np
    height, width, channels = image.shape
    mask = np.zeros((height, width, channels), dtype=np.uint8)
    
    color = mapping[0]   
    
    for k in mapping:
        # Get all indices for current class
        idx = (pred==torch.tensor(k, dtype=torch.uint8))
        idx_np = (y_pred==k)[0]
        # color = mapping[k]
        mask[idx_np] = (mapping[k])
    
    cv2.imwrite(destination_mask+'/'+name, mask)
    '''
    dst = cv2.addWeighted(img[0,...].transpose(1,2,0), 1, mask.transpose(1,2,0), 0.75, 0)
    # Plot the input image, ground truth and the predicted output
    plt.figure(figsize=(20,20));
    plt.subplot(131);
    plt.imshow(img[0,...].transpose(1,2,0));
    plt.title('Image')
    plt.axis('off');
    plt.subplot(132);
    plt.imshow(mask);
    plt.title('Ground Truth Mask')
    plt.axis('off');
    plt.subplot(133);
    #t =a['out'].cpu().detach().numpy()[0]
    #plt.imshow(a['out'].cpu().detach().numpy()[0][0]);
    plt.imshow(dst);
    plt.title('Mask Overlay')
    plt.axis('off');
    plt.savefig(destination_combined + '/'+name,bbox_inches='tight')
    
    
    
import os 
from tqdm import tqdm   
# Load the trained model 
model = torch.load(f'weights_5.pt', map_location=torch.device('cpu'))
# Set the model to evaluate mode
model.eval()

# # Read the log file using pandas into a dataframe
df = pd.read_csv(f'log.csv')

# # Plot all the values with respect to the epochs
df.plot(x='epoch',figsize=(15,8));

# print(df[['Train_f1_score','Test_f1_score']].max())

source_dir = 'Test_masks/'
destination_mask = 'mask_ground_truth_Test/'
destination_combined = 'combined_ground_truth_Test'
for image_name in tqdm(os.listdir(source_dir)):
    print(image_name)
    image_path = source_dir + image_name
    generate_images(model, image_path, image_name, destination_mask, destination_combined)