# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:00:49 2020

@author: Eric Bianchi
"""
import numpy as np
import cv2
import os
import sys

input_directory = './inpainting_combined/'
#  input_directory = './combined_images/'

name = 'image_list'
f = open("./" + name + ".txt", "w")

for im in os.listdir(input_directory):
    f.write(input_directory+im+'\n')
f.close()
    