import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors
import PIL
import os
import pdb
import theano
import pickle

def GetImage(path):
    path = os.path.expanduser(path)
    img = np.array(Image.open(path).convert('RGBA'))
    mask = (img[:,:,3]==0)
    img[:,:,:4][mask] = [255,255,255,255]
    # image = Image.fromarray(img).convert('RGB').resize((64,106),Image.ANTIALIAS)
    image = Image.fromarray(img).convert('RGB').resize((128,192),Image.ANTIALIAS)
    nprgb = np.asarray(image,'float32')/255.
    hsvnp = matplotlib.colors.rgb_to_hsv(nprgb)
    return hsvnp.reshape([-1])

def GetData(path="~/Data/charactersDS/"):

    # trX=np.zeros([700,106*64*3],dtype='float32')
    # vaX=np.zeros([200,106*64*3],dtype='float32')
    path = os.path.expanduser(path)
    trX=np.zeros([700,192*128*3],dtype='float32')
    vaX=np.zeros([200,192*128*3],dtype='float32')
   
    countertr=0;counterva=0;counter=0
    lstFiles=os.listdir(path)
    lstFiles=np.sort(os.listdir(path))
    for file in lstFiles:
        if (counter+1)%5==0:
            vaX[counterva,:] = GetImage(path+file)
            counterva+=1
        else:#tr
            trX[countertr,:] = GetImage(path+file)
            countertr+=1
        counter +=1
    trX=trX[0:countertr]
    vaX=vaX[0:counterva]
    
    train_set, valid_set = [trX, vaX]
    
    def shared_dataset(data_x, borrow=True):
        shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        return shared_x

    valid_set_x  = shared_dataset(valid_set)
    train_set_x  = shared_dataset(train_set)
    rval = [train_set_x,valid_set_x]
    return rval
    

