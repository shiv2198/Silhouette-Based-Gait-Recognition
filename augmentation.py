import cv2
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf

count = 31
walk_angle = '018'
parent_dir = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/train/'+str(walk_angle)+'/'
#parent_dir = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+str(walk_angle)+'/'+'GEI/'

for gender in os.listdir(parent_dir):

    for i in os.listdir(parent_dir+gender):
        try:

            img = cv2.imread(parent_dir+gender+'/'+i, cv2.IMREAD_GRAYSCALE)
            print(img)
            flip = np.fliplr(img)
            image = Image.fromarray(flip)
            image.save(parent_dir+gender+'/'+str(count)+'.png')
            count +=1





        except Exception as e:
            print(str(e))











