# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:10:15 2020

@author: Eric Bianchi
"""
import sys

sys.path.insert(0, 'S://Research/Python/general_utils/')

from image_utils import extension_change

source_image_folder = 'S://adv_cv_final/OLD/Unusable/'
destination = 'S://adv_cv_final/unusable_jpeg/'
extension = 'jpeg'
extension_change(source_image_folder, destination, extension)