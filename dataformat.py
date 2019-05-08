
#encoding='utf-8'

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



CIRFA_DIR=os.path.abspath(os.path.join(os.getcwd(), "../cifar-10-batches-py"))

#print (os.listdir(CIRFA_DIR))
with open(os.path.join(CIRFA_DIR,'data_batch_1'),'rb') as f:
    #data=pickle.load(f.read())
    data=pickle.load(f,encoding='bytes')
    print(type(data))
    print(data.keys())
    print(type(data[b'data']))
    print(type(data[b'labels']))
    print(data[b'data'].shape)
    print(data[b'data'][0:2])
    print(data[b'labels'])
    print((data[b'batch_label']))
    print(data[b'filenames'][0:2])
    #32*32=1024*3=3072
    image_arr=data[b'data'][100]
    image_arr=image_arr.reshape(3,32,32)
    image_arr=image_arr.transpose(1,2,0)

    imshow(image_arr)