# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:51:00 2020

@author: Eric Bianchi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:30:23 2020

@author: Eric Bianchi
"""
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10

import os
import glob
import numpy as np
from scipy.misc import imread
from tqdm import tqdm


import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

mu_list = []
sigma_list = []
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

LAN_280 = './LAN_280/'
gen_model_194 = './gen_model_194/'

data_list = [gen_model_194, LAN_280]
 # 'StyleGAN2_general', 'annotated_images_jpeg', 'FFHQ', 
data_list_name = ['gen_model_194', 'LAN_280']

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

def calc_muSigma(model, images):
    with tf.device('gpu'):
        print('calcuating acitvations')
        # calculate activations
        activation = model.predict(images)
        print('acitvations: complete')
        mu1, sigma1 = activation.mean(axis=0), cov(activation, rowvar=False)

        return mu1, sigma1

def calc_fid(mu1, sigma1, mu2, sigma2):
  # calculate sum squared difference between means
  print('ssdiff')
  ssdiff = numpy.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  print('covmean')
  covmean = sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
        covmean = covmean.real
  # calculate score
  fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

# calculate activations
import os
from numpy import asarray
from numpy import save
for i in range(0,len(data_list)):

  name = data_list_name[i]
  print('finding activations for: ' + name)
  print('path: ' + data_list[i])

  # loads all images into memory (this might require a lot of RAM!)
  print("load train images..", end=" " , flush=True)
  image_list = glob.glob(os.path.join(data_list[i], '*.jpg'))

  image_list = np.array([imread(str(fn)).astype(np.float32) for fn in tqdm(image_list)])
  print("%d images found and loaded" % len(image_list))

  # resize images
  images1 = scale_images(image_list, (299,299,3))
  print('Scaled', images1.shape)
  # pre-process images
  images1 = preprocess_input(images1)
  print('pre-processed images')

  mu, sigma = calc_muSigma(model, images1)

  save(data_list_name[i]+'_mu.npy', mu)
  save(data_list_name[i]+'_sigma.npy', sigma)

  print(10*'------')