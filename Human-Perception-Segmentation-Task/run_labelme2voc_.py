# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:25:00 2020

@author: Eric Bianchi
"""

import sys
sys.path.insert(0, "S://Research/Python/3_annotate_data/")

from labelme2voc_ import createMasks
sys.path.insert(0, "S://Research/Python/general_utils/")
# from image_utils import extension_change

# C:\Users\Eric Bianchi\Desktop\images
# createMasks(input_dir, output_dir, label_txt_file, imivz=True)
input_dir = 'S://adv_cv_final/occlusion_study/'
output_dir = 'S://adv_cv_final/occlusion_voc/'
label_txt_file = 'S://adv_cv_final/class_names.txt'

# (source_image_folder, destination, extension):
# extension_change(input_dir, input_dir_, '.jpeg')

createMasks(input_dir, output_dir, label_txt_file)