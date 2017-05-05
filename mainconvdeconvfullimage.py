import numpy as np
import time
import os
from VAEConv import VAEConvDec
import pickle
import gzip
import pdb
from dataloader import *
from optimizer import AdamOptimizer,RMSPROP,GDA
import pdb
import matplotlib.pyplot as plt


theano.config.dnn.conv.algo_bwd_filter="deterministic"
theano.config.dnn.conv.algo_bwd_data  ="deterministic"




#hu_encoder (ks,fw,fh,poolw,poolH)
conf={
    'lamdaKL':1,
    'hu_encoder':[(50,3,3,3,3),(60,3,3,2,2)],
    'n_latent':36,
    'hu_decoder':[(20,5,5),200],
    'n_epochs':100,
    'batch_size':20,
    'learning_rate':5e-3,
    'l2':0,
    'optim':'rmsprop',
     #adam
    'b1':0.5,
    'b2':0.99,
     #rmsprop
    'decay_rate':0.9
}


lamdaKL = conf['lamdaKL']
hu_encoder = conf['hu_encoder']
hu_decoder = conf['hu_decoder']
n_latent = conf['n_latent']
n_epochs = conf['n_epochs']

batch_size=conf['batch_size']
learning_rate = conf['learning_rate']

print ("Loading  data")
x_train,x_valid = GetData()

print ("Data was loaded")


sampleSize,featureSize = x_train.get_value().shape
validSize=x_valid.get_value().shape[0]
# pdb.set_trace()
path = os.path.expanduser("~/ModelOut/")

print ("instantiating model")
rng = np.random.RandomState(42)
model = VAEConvDec(conf,rng,featureSize, hu_encoder, hu_decoder, n_latent, lamdaKL,
                  rows=192,cols=128,inpChannels=3)
# pdb.set_trace()
if conf['optim']=='rmsprop':
    optim = RMSPROP(model.params,learning_rate, batch_size=batch_size,decay_rate=conf['decay_rate'],l2=conf['l2'])
elif conf['optim']=='gda':
    optim = GDA(model.params,learning_rate, batch_size=batch_size,l2=conf['l2'])
else:
    optim = AdamOptimizer(model.params,learning_rate, sampleSize, 
        b1=conf['b1'], b2=conf['b2'], batch_size=batch_size,l2=conf['l2'])

# pdb.set_trace() 
tf_update= model.getUpdates(x_train,optim) #[batch,epoch,lr ]


batch_order = np.arange(int(sampleSize / batch_size))
epoch = 0
LB_list = []


# model.load_parameters(path,adam)

genimage = model.decodeFromPrior()
lstupTraceForPlot=[]
if __name__ == "__main__":
    print ("iterating")
    bestValidLB = -np.inf
    trLB=0
    clrDec = 0
    lrPatience=5
    while epoch < n_epochs:
        if epoch==10:
            model.running_average_factor.set_value(0.5)
            # pdb.set_trace()
        elif epoch == 20:
            model.running_average_factor.set_value(0.75)
        elif epoch == 25:
            model.running_average_factor.set_value(0.95)

        epoch += 1
        rng.shuffle(batch_order)
        LB = 0.

        for batch in batch_order:
            # pdb.set_trace()
            batch_LB = tf_update(batch, epoch,optim.learning_rate)
            # lstupTraceForPlot.append(np.asarray(batch_LB[3]))
            # lstupTraceForPlot.append([np.linalg.norm(item) for item in batch_LB[1:]])
            LB += batch_LB[0]
        
        LB /= len(batch_order)

        model.trainmode.set_value(0)
        valid_LB,stderror = model.likelihood(x_valid.get_value())
        print ("Epoch {0} finished. TrLB: {1}, VaLB {2}".format(epoch, LB, valid_LB))
        
        
        if valid_LB > bestValidLB:
            b_valid_trlb,trstderror = model.likelihood(x_train.get_value())
            print("------> Better results acheived. Tr: {0} , Va: {1} , VaSTDE: {2}".format(
                                                b_valid_trlb, valid_LB,  stderror/np.sqrt(validSize)))
            bestValidLB = valid_LB
            model.save_parameters(path,optim)
            clrDec=0
        else:
            clrDec +=1
            if clrDec > lrPatience:
                clrDec = 0
                optim.learning_rate /=2.
                print(">>>>>>Learning Rate was decreased<<<<")
       
        

        model.trainmode.set_value(1)

    print( "\n\n ********* BEST TR: {0}, BEST VAL: {1} ".format(
                                b_valid_trlb,bestValidLB))
   
    pickle.dump(lstupTraceForPlot,open(path+"/up.pkl",'wb'))
    
    

# import matplotlib.pyplot as plt

# def GetImageToView(nphsvim):
#     preToRGB= matplotlib.colors.hsv_to_rgb(nphsvim.reshape([192,128,3]))*255
#     preToRGB[preToRGB<0] = 0
#     preToRGB[preToRGB>255] = 255
#     preToRGB2 = np.asarray(np.round(preToRGB),'uint8')
#     return Image.fromarray(preToRGB2)

# genimage = model.decodeFromPrior(100)

# model.trainmode.set_value(0.)
# model.learningMode.set_value(0.)

# mu,sigma,prior = genimage()

# for i in range(100):
#     img = GetImageToView(mu[i])
#     img.save("/home/hani/Data/gen/characterDS/"+str(i)+".bmp")

# model.learningMode.set_value(1.)
# model.trainmode.set_value(1.)
