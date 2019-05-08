
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os,sys
from os import *
import skimage.io as io
from PIL import Image



#method to load testing images
def testGenerator(test_path,general_name,f,t,target_size = (256,256)):
    for i in range(f,t+1):
        img = io.imread(os.path.join(test_path,general_name%i))
        img = img / 255
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
#method to save predicted images
def save_result(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"greyscale%d_gt.png"%(i+1)),img)
def load_image(infilename):
    return mpimg.imread(infilename)

def resize_image(filename, origin_dir, new_dir, new_size):
    img = Image.open(origin_dir+filename)
    img = img.resize(new_size,Image.NEAREST)
    img.save(new_dir+filename)