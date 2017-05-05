import math
import numpy as np
import theano
import theano.tensor as T
import pdb
import pickle
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from skimage import io
from PIL import Image
import itertools
import matplotlib.colors
import PIL
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def ortho_weight(ran,ndim):
    # ran = np.random.RandomState(123)
    # W = ran.randn(ndim, ndim)
    W = ran.uniform(    low=-np.sqrt(6. / (ndim*2)),
                        high=np.sqrt(6. / (ndim*2)),
                        size=(ndim, ndim)
                    ).astype(theano.config.floatX)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)



def getActivation(txtactivation,x):
    if txtactivation == "tanh":
        return T.tanh(x)
    elif txtactivation == "sigmoid":
        return T.nnet.sigmoid(x)
    elif txtactivation == "rectifier":
        return T.nnet.relu(x)
    elif txtactivation == "linear":
        return x

class HiddenLayer(object):
    def __init__(self, rng, trainmode,running_average_factor,
                        input, n_in, n_out, name = 'hidden',txtactivation="rectifier",
                        W=None, b=None,orth=1):

        self.input = input
        self.txtactivation = txtactivation

        self.trainmode = trainmode
        self.running_average_factor = running_average_factor   

        if W is None:
            # pdb.set_trace()
            if n_out<=n_in and orth==1:
                W_values = ortho_weight(rng,n_in)[:,0:n_out]
                
                print("ORTHOGONAL WEIGHTS WERE USED...")
            else:
                # W_values = rng.normal(0, 0.01, (n_in, n_out)).astype(theano.config.floatX)
                W_values = rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ).astype(theano.config.floatX)
            # W_values = rng.uniform(
            #         low=-np.sqrt(6. / (n_in + n_out)),
            #         high=np.sqrt(6. / (n_in + n_out)),
            #         size=(n_in, n_out)
            #     ).astype(theano.config.floatX)

            W = theano.shared(value=W_values*4, name='W_'+name, borrow=True)
        
        self.W = W


        # self.gamma = theano.shared(np.asarray(rng.uniform(low=-1.0/math.sqrt(n_out), 
        #                 high=1.0/math.sqrt(n_out), size=(n_out)), dtype=theano.config.floatX),
        #                  name='gamma_'+name, borrow=True)
        self.gamma = theano.shared(np.asarray(rng.uniform(low=-np.sqrt(6. / ( n_out)), 
                        high=np.sqrt(6. / ( n_out)), size=(n_out)), dtype=theano.config.floatX),
                         name='gamma_'+name, borrow=True)
        self.beta = theano.shared(np.zeros((n_out), dtype=theano.config.floatX), name='beta_'+name, borrow=True)
        
        self.linear=T.dot(input, self.W)

        self.new_running_mean = theano.shared(np.zeros((n_out), dtype=theano.config.floatX), 
                                    name='rmean_'+name, borrow=True)
        self.new_running_var  = theano.shared(np.zeros((n_out), dtype=theano.config.floatX), 
                                    name='rvar_'+name, borrow=True)
        mean= T.mean(self.linear,axis=0)
        var = T.var (self.linear,axis=0)
        self.bnTr = (self.linear - mean)/(T.sqrt(var+0.0001))
        self.bnTe = (self.linear - self.new_running_mean)/(T.sqrt(self.new_running_var+0.0001))

        self.lbnOut = T.switch(self.trainmode,
                                self.bnTr, 
                                self.bnTe)

        self.actInput = self.gamma *self.lbnOut + self.beta
        # self.actInput = self.lbnOut + self.beta
        self.output = getActivation(self.txtactivation,self.actInput)
        # parameters of the model
        self.params = [self.W, self.gamma, self.beta]
        # self.params = [self.W, self.beta]
        # self.raupdates = OrderedDict()
        # self.raupdates.update({'rmean_'+name:self.new_running_mean*self.running_average_factor+mean*(1-self.running_average_factor)})
        # self.raupdates.update({'rvar_' +name:self.new_running_var*self.running_average_factor+var*(1-self.running_average_factor)})
        self.raupdates=[(self.new_running_mean, 
                            self.new_running_mean*self.running_average_factor+mean*(1-self.running_average_factor)),
                      (self.new_running_var, 
                            self.new_running_var*self.running_average_factor+var*(1-self.running_average_factor))]
        
        self.raparams = [self.new_running_mean,self.new_running_var]

    def load(self,obj):
        self.W = obj.W
        self.gamma = obj.gamma
        self.beta  = obj.beta
        self.txtactivation = obj.txtactivation
        self.output = getActivation(self.txtactivation,self.lbnOut)


class GaussianLayer(object):
    def __init__(self, rng, input, n_in, n_out, name = 'gaussianhidden',
                        W_mu=None,W_sig=None, b_mu=None,b_sig= None,orth=1):
       
        self.input = input
        self.rng = rng
        if W_mu is None:
            # W_mu  = rng.normal(0, 0.01, (n_in, n_out)).astype(theano.config.floatX)
            if n_out<=n_in and orth==1:
                W_mu = ortho_weight(rng,n_in)[:,0:n_out]
                # pdb.set_trace()
                print("ORTHOGONAL WEIGHTS WERE USED...")
            else:

                W_mu = rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ).astype(theano.config.floatX)
            # W_mu = rng.uniform(
            #             low=-np.sqrt(6. / (n_in + n_out)),
            #             high=np.sqrt(6. / (n_in + n_out)),
            #             size=(n_in, n_out)
            #         ).astype(theano.config.floatX)
            W_mu = theano.shared(value=W_mu*4, name='Wmu_'+name, borrow=True)

        if W_sig is None:
            # W_sig = rng.normal(0, 0.01, (n_in, n_out)).astype(theano.config.floatX)
            if n_out<=n_in and orth==1:
                W_sig = ortho_weight(rng,n_in)[:,0:n_out]
                print("ORTHOGONAL WEIGHTS WERE USED...")
            else:
                W_sig = rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ).astype(theano.config.floatX)
            # W_sig = rng.uniform(  
            #             low=-np.sqrt(6. / (n_in + n_out)),
            #             high=np.sqrt(6. / (n_in + n_out)),
            #             size=(n_in, n_out)
            #         ).astype(theano.config.floatX)
            W_sig = theano.shared(value=W_sig*4, name='Wsig_'+name, borrow=True)

        if b_mu is None:
            b_mu  = np.zeros((n_out,), dtype=theano.config.floatX)
            b_mu = theano.shared(value=b_mu, name='bmu_'+name, borrow=True)
        if b_sig is None:    
            b_sig  = np.zeros((n_out,), dtype=theano.config.floatX)
            b_sig = theano.shared(value=b_sig, name='bsig_'+name, borrow=True)

        self.W_mu  = W_mu  ; self.b_mu  = b_mu
        self.W_sig = W_sig ; self.b_sig = b_sig
        self.output_mu   = T.dot(input, self.W_mu)  + self.b_mu
        self.output_sig  = T.dot(input, self.W_sig) + self.b_sig

        self.params = [self.W_mu, self.W_sig, self.b_mu, self.b_sig]
        self.raupdates = []#OrderedDict()
        self.raparams=[]

        seed = 42 #42
        if "gpu" in theano.config.device:
            self.srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            self.srng = T.shared_randomstreams.RandomStreams(seed=seed)
        # self.srng = RandomStreams(seed=seed) 
    def sampler(self):
        
        eps = self.srng.normal(self.output_mu.shape)
        # z = self.output_mu + T.exp(0.5 * T.minimum(self.output_sig,-100)) * eps
        z = self.output_mu + T.exp(0.5 * self.output_sig) * eps
        # z = self.output_mu 
        return z

    def logpxz(self,x):
        logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * self.output_sig) -
                      0.5 * ((x - self.output_mu)**2 / T.exp(self.output_sig))).sum(axis=1)
        return logpxz

    def load(self,obj):
        self.W_mu = obj.W_mu
        self.b_mu = obj.b_mu
        self.W_sig = obj.W_sig
        self.b_sig = obj.b_sig

class LeNetConvPoolLayer(object):

    def __init__(self, rng, trainmode,running_average_factor,input, filter_shape, image_shape, poolsize=(2, 2),name='convlayer'):
        self.input = input
        self.trainmode =trainmode
        self.running_average_factor=running_average_factor
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),name='W_'+name,borrow=True
        )
        # b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        # self.b = theano.shared(value=b_values, name='b_'+name,borrow=True)
        self.conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        # pool each feature map individually, using maxpooling
        self.pooled_out = pool.pool_2d(
            input=self.conv_out,
            ws=poolsize,
            ignore_border=True
        )
        
        self.gamma = theano.shared(np.asarray(rng.uniform(low=-np.sqrt(6. / ( filter_shape[0])), 
                        high=np.sqrt(6. / ( filter_shape[0])), size=(filter_shape[0],)), dtype=theano.config.floatX),
                         name='gamma_'+name, borrow=True)
        self.beta = theano.shared(np.zeros((filter_shape[0],), dtype=theano.config.floatX), name='beta_'+name, borrow=True)
        
        
        self.new_running_mean = theano.shared(np.zeros((filter_shape[0]), dtype=theano.config.floatX), 
                                    name='rmean_'+name, borrow=True)
        self.new_running_var  = theano.shared(np.zeros((filter_shape[0]), dtype=theano.config.floatX), 
                                    name='rvar_'+name, borrow=True)
        
        mean = T.mean(self.pooled_out, axis=[0, 2, 3])#.dimshuffle('x', 0, 'x', 'x')
        var  = T.mean(T.sqr(self.pooled_out - mean.dimshuffle('x', 0, 'x', 'x')), axis=[0, 2, 3])#.dimshuffle('x', 0, 'x', 'x')
        
        self.bnTr = (self.pooled_out - mean.dimshuffle('x', 0, 'x', 'x'))/(T.sqrt(var.dimshuffle('x', 0, 'x', 'x')+0.0001))
        self.bnTe = (self.pooled_out - self.new_running_mean.dimshuffle('x', 0, 'x', 'x'))/(T.sqrt(self.new_running_var.dimshuffle('x', 0, 'x', 'x')+0.0001))
        
        self.lbnOut = T.switch(self.trainmode,
                                self.bnTr, 
                                self.bnTe)
        self.actInput = self.gamma.dimshuffle('x', 0, 'x', 'x') *self.lbnOut + self.beta.dimshuffle('x', 0, 'x', 'x')
        # activation = lambda x: T.switch(T.gt(x,6), 1+T.mul(x,0.1) ,
        #                                   T.switch(T.lt(x,-6),T.mul(x,0.1),(1/12.)*x+0.5))
        
        self.output = T.nnet.relu(self.actInput)
        # self.output = activation(self.actInput)
        self.params = [self.W, self.gamma, self.beta]
        # self.params = [self.W, self.beta]
     
        self.raupdates=[(self.new_running_mean, 
                            self.new_running_mean*self.running_average_factor+mean*(1-self.running_average_factor)),
                      (self.new_running_var, 
                            self.new_running_var*self.running_average_factor+var*(1-self.running_average_factor))]
        
        self.raparams = [self.new_running_mean,self.new_running_var]


    # def loadParams(path):
    #     pickle.dump({name: p.get_value() for name, p in odic_params.items()}, open(path + "/params.pkl", "wb"))
    #     pickle.dump({p.name: p.get_value() for p in hlayer.raparams}, open(path + "/raparams.pkl", "wb"))
    #     pickle.dump(conf,open(path+"/conf.pkl","wb"))

class DeconvLayer(object):

    def __init__(self, rng, trainmode,running_average_factor,input,
        inpChannels,inpW,inpH,ks,fw,fh,name='decconvlayer'):
        self.input = input
        self.inpChannels=inpChannels
        self.inpW=inpW
        self.inpH=inpH
        self.ks=ks
        self.fw=fw
        self.fh=fh

        self.trainmode =trainmode
        self.running_average_factor=running_average_factor
        fan_in = inpChannels*inpW*inpH
        fan_out = ks*fw*fh
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=(inpChannels ,ks,fw,fh)),
                dtype=theano.config.floatX
            ),name='W_'+name,borrow=True
        )
        self.outshape= (None,ks,inpW+fw-1,inpH+fh-1)

        self.deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            output_grad = self.input,
            filters=self.W,
            input_shape= self.outshape,
            border_mode=(0,0), 
            subsample=(1,1)
        )
       
        
        self.gamma = theano.shared(np.asarray(rng.uniform(low=-np.sqrt(6. / ks), 
                        high=np.sqrt(6. / ks), size=(ks,)), dtype=theano.config.floatX),
                         name='gamma_'+name, borrow=True)
        self.beta = theano.shared(np.zeros((ks,), dtype=theano.config.floatX), name='beta_'+name, borrow=True)
        
        
        self.new_running_mean = theano.shared(np.zeros((ks), dtype=theano.config.floatX), 
                                    name='rmean_'+name, borrow=True)
        self.new_running_var  = theano.shared(np.zeros((ks), dtype=theano.config.floatX), 
                                    name='rvar_'+name, borrow=True)
        
        mean = T.mean(self.deconv_out, axis=[0, 2, 3])#.dimshuffle('x', 0, 'x', 'x')
        var  = T.mean(T.sqr(self.deconv_out - mean.dimshuffle('x', 0, 'x', 'x')), axis=[0, 2, 3])#.dimshuffle('x', 0, 'x', 'x')
        
        self.bnTr = (self.deconv_out - mean.dimshuffle('x', 0, 'x', 'x'))/(T.sqrt(var.dimshuffle('x', 0, 'x', 'x')+0.0001))
        self.bnTe = (self.deconv_out - self.new_running_mean.dimshuffle('x', 0, 'x', 'x'))/(T.sqrt(self.new_running_var.dimshuffle('x', 0, 'x', 'x')+0.0001))
        
        self.lbnOut = T.switch(self.trainmode,
                                self.bnTr, 
                                self.bnTe)
        self.actInput = self.gamma.dimshuffle('x', 0, 'x', 'x') *self.lbnOut + self.beta.dimshuffle('x', 0, 'x', 'x')
        # activation = lambda x: T.switch(T.gt(x,6), 1+T.mul(x,0.1) ,
        #                                   T.switch(T.lt(x,-6),T.mul(x,0.1),(1/12.)*x+0.5))
        
        self.output = T.nnet.relu(self.actInput)
        # self.output = activation(self.actInput)
        self.params = [self.W, self.gamma, self.beta]
        # self.params = [self.W, self.beta]
     
        self.raupdates=[(self.new_running_mean, 
                            self.new_running_mean*self.running_average_factor+mean*(1-self.running_average_factor)),
                      (self.new_running_var, 
                            self.new_running_var*self.running_average_factor+var*(1-self.running_average_factor))]
        
        self.raparams = [self.new_running_mean,self.new_running_var]            



class VAEConvDec:
    
    def __init__(self,conf,rng, n_in, lst_hu_encoder, lst_hu_decoder, n_latent,lamdaKL
        ,runningaverage=0.4,generaterSeed=1234,rows=48,cols=32,inpChannels=1):

        self.conf=conf
        self.layers=[]
        self.lst_hu_encoder = lst_hu_encoder
        self.lst_hu_decoder = lst_hu_decoder
        self.n_latent = n_latent
        self.lamdaKL = lamdaKL
        self.n_in = n_in

        self.x = T.matrix("x")
        self.epoch = T.scalar("epoch")
        self.batch = T.iscalar('batch')
        self.trainmode = theano.shared(np.asarray(1., dtype=theano.config.floatX))
        self.learningMode = theano.shared(np.asarray(1., dtype=theano.config.floatX)) #0 means generation for images
        self.running_average_factor=theano.shared(np.asarray(runningaverage, dtype=theano.config.floatX))

        currentInputSize= n_in
        inputTensor =T.reshape(self.x,(self.x.shape[0], inpChannels, rows, cols))
        for i in range(len(self.lst_hu_encoder)):
            if type(self.lst_hu_encoder[i]) is tuple:
                print(rows,cols)
                self.layers.append(
                    LeNetConvPoolLayer(rng,trainmode=self.trainmode,running_average_factor=self.running_average_factor, 
                            input=inputTensor,
                            image_shape=(None, inpChannels, rows, cols),
                            filter_shape=(self.lst_hu_encoder[i][0],inpChannels,self.lst_hu_encoder[i][1],self.lst_hu_encoder[i][2]),
                            poolsize=(self.lst_hu_encoder[i][3], self.lst_hu_encoder[i][4]),name='encoder_convlayer'+str(i))
                    )

                rows  = int((rows -self.lst_hu_encoder[i][1]+1)/self.lst_hu_encoder[i][3])
                cols  = int((cols-self.lst_hu_encoder[i][2]+1)/self.lst_hu_encoder[i][4])
                inpChannels = self.lst_hu_encoder[i][0]
                inputTensor = self.layers[-1].output
            else:
                print(rows,cols)
                if (i-1)>=0 and type(self.lst_hu_encoder[i-1]) is tuple:
                    inputTensor = inputTensor.flatten(2)
                    currentInputSize = rows*cols*lst_hu_encoder[i-1][0]
                    
                self.layers.append(
                    HiddenLayer(rng=rng,trainmode=self.trainmode,running_average_factor=self.running_average_factor,
                                 input=inputTensor, n_in=currentInputSize, n_out=self.lst_hu_encoder[i],
                                name = 'encoder'+str(i),txtactivation="rectifier",orth=0))
                currentInputSize = lst_hu_encoder[i]
                inputTensor = self.layers[-1].output
                            
        if type(self.lst_hu_encoder[-1]) is tuple:
            # print(rows,cols)
            currentInputSize = rows*cols*lst_hu_encoder[-1][0]
            inputTensor =inputTensor.flatten(2) 
        print(rows,cols)
        self.layers.append(
            GaussianLayer(rng=rng, input=inputTensor, n_in=currentInputSize, n_out=self.n_latent, name = 'latent',orth=0))
        
        self.KLD = 0.5 * T.sum(1 + self.layers[-1].output_sig - self.layers[-1].output_mu**2 - T.exp(self.layers[-1].output_sig), axis=1)
        
        # self.z =  self.layers[-1].sampler()
        if "gpu" in theano.config.device:
            self.genrng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=generaterSeed)
        else:
            self.genrng = T.shared_randomstreams.RandomStreams(seed=generaterSeed)

        zgivenx = self.layers[-1].sampler()
        self.randFromPrior = self.genrng.normal(zgivenx.shape,avg=0.0, std=3.0)
        
        sqrts = int(np.ceil(np.sqrt(n_latent)))
        self.tmp1=T.zeros([zgivenx.shape[0],sqrts**2 - n_latent])
        self.tmp2=zgivenx
        
        if type(self.lst_hu_decoder[0]) is tuple:
            if (sqrts**2 - n_latent)>0:
                zgivenxzropad = T.concatenate([zgivenx,T.zeros([zgivenx.shape[0],sqrts**2 - n_latent])],axis=1)
                self.randFromPrior = T.concatenate([self.randFromPrior,T.zeros([self.randFromPrior.shape[0],sqrts**2 - n_latent])],axis=1)
            else:
                zgivenxzropad = zgivenx
            zgivenxzropad = zgivenxzropad.reshape((-1,1,sqrts,sqrts))
            self.randFromPrior = self.randFromPrior.reshape((-1,1,sqrts,sqrts))
        else:
            zgivenxzropad = zgivenx
        #check if it is LEARNING MODE OR GENERATING IMAGES MODE
        
        self.z = T.switch(   self.learningMode,
                             zgivenxzropad,
                             self.randFromPrior)
        
        currentInputSize = self.n_latent
        inputTensor = self.z  
        inpChannels=1;inpW=rows=sqrts;inpH=cols=sqrts;
        print(inpChannels,inpW,inpH)
        for i in range(len(self.lst_hu_decoder)):
            if type(self.lst_hu_decoder[i]) is tuple:
                if len(self.lst_hu_decoder[i])==3:
                    self.layers.append(
                        DeconvLayer(rng=rng,trainmode=self.trainmode,running_average_factor=self.running_average_factor,
                                        input=inputTensor,
                                        inpChannels=inpChannels,inpW=inpW,inpH=inpH,
                                        ks=self.lst_hu_decoder[i][0],
                                        fw=self.lst_hu_decoder[i][1],
                                        fh=self.lst_hu_decoder[i][2],
                                        name='decoder_deconv'+str(i))
                    )
                    inputTensor = self.layers[-1].output  
                    inpChannels = self.lst_hu_decoder[i][0];
                    inpW=self.layers[-1].outshape[2]
                    inpH=self.layers[-1].outshape[3]
                    #for LeNet Conv
                    rows = self.layers[-1].outshape[2]
                    cols = self.layers[-1].outshape[3]
                    print(inpChannels,inpW,inpH)
                else:
                    
                    self.layers.append(
                        LeNetConvPoolLayer(rng,trainmode=self.trainmode,running_average_factor=self.running_average_factor, 
                            input=inputTensor,
                            image_shape=(None, inpChannels, rows, cols),
                            filter_shape=(self.lst_hu_decoder[i][0],inpChannels,self.lst_hu_decoder[i][1],self.lst_hu_decoder[i][2]),
                            poolsize=(self.lst_hu_decoder[i][3], self.lst_hu_decoder[i][4]),
                            name='decoder_convlayer'+str(i))
                    )

                    rows  = int((rows -self.lst_hu_decoder[i][1]+1)/self.lst_hu_decoder[i][3])
                    cols  = int((cols -self.lst_hu_decoder[i][2]+1) /self.lst_hu_decoder[i][4])
                    inpW=rows
                    inpH=cols
                    inpChannels = self.lst_hu_decoder[i][0]
                    inputTensor = self.layers[-1].output
                    print(rows,cols)
               
            else:
                if (i-1)>=0 and type(self.lst_hu_decoder[i-1]) is tuple:
                    inputTensor = inputTensor.flatten(2)
                    currentInputSize = inpW*inpH*inpChannels

                self.layers.append(
                    HiddenLayer(rng=rng, trainmode=self.trainmode,running_average_factor=self.running_average_factor,
                                input=inputTensor, n_in=currentInputSize, n_out=self.lst_hu_decoder[i],
                                 name = 'decoder'+str(i),txtactivation="rectifier",orth=0))

                currentInputSize = lst_hu_decoder[i]
                inputTensor = self.layers[-1].output


        # pdb.set_trace()
        if type(self.lst_hu_decoder[-1]) is tuple:
            inputTensor = inputTensor.flatten(2)
            currentInputSize = inpW*inpH*inpChannels
        self.layers.append(
            GaussianLayer(rng=rng, input=inputTensor, n_in=currentInputSize, n_out=n_in, name = 'output',orth=0))
        self.params = OrderedDict()
        self.raupdates = []
        self.raparams = OrderedDict()
        for l in self.layers:
            for p in l.params:
                self.params.update({p.name:p})
            for p in l.raparams:
                self.raparams.update({p.name:p})
            self.raupdates += l.raupdates
             

        self.cost = T.mean(self.layers[-1].logpxz(self.x) + self.lamdaKL* self.KLD)
        stderror = T.std(self.layers[-1].logpxz(self.x) + self.lamdaKL* self.KLD)
        self.likelihood = theano.function([self.x], [self.cost,stderror])
        
        self.getKLLogPxz   = theano.function([self.x],[self.KLD, self.layers[-1].logpxz(self.x)])
        self.getOutput= theano.function([self.x],[self.layers[-1].output_mu, self.layers[-1].output_sig ])
        self.getzvalues = theano.function([self.x],zgivenx )
        
        
    def decodeFromPrior(self,sampleSize=10):
        # pdb.set_trace()
        generateImage = theano.function([],[ self.layers[-1].output_mu, self.layers[-1].output_sig ,self.randFromPrior],
            givens={
                        self.x: np.zeros([sampleSize,self.n_in ],theano.config.floatX)
        })
        return generateImage

    def getGrads(self,optimizer):

        gradients = T.grad(self.cost, list(self.params.values()))
        # Adam implemented as updates
        updates, uptrace = optimizer.get_updates(gradients, self.epoch )
        return [updates,uptrace,gradients]

    def getUpdates(self,x_train,optimizer):
        updates,uptrace,g = self.getGrads(optimizer)
        lstupdates= list(updates.items())
        output= list(itertools.chain(*[[self.cost],uptrace]))
        # pdb.set_trace()
        update = theano.function([self.batch, self.epoch, optimizer.lr ],output , updates=lstupdates+self.raupdates, 
            givens={
                                        self.x: x_train[self.batch*optimizer.batch_size:(self.batch+1)*optimizer.batch_size, :]
        })

        return update


    def save_parameters(self, path,optimizer):
        """Saves all the parameters in a way they can be retrieved later"""
        pickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        pickle.dump({name: p.get_value() for name, p in self.raparams.items()}, open(path + "/raparams.pkl", "wb"))
        pickle.dump(self.conf,open(path+"/conf.pkl","wb"))
        # pickle.dump({name: m.get_value() for name, m in optimizer.m.items()}, open(path + "/m.pkl", "wb"))
        # pickle.dump({name: v.get_value() for name, v in optimizer.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path,optimizer):
        """Load the variables in a shared variable safe way"""
        self.conf=pickle.load(open(path +"conf.pkl","rb"))
        p_list = pickle.load(open(path + "params.pkl", "rb"))
        ra_list = pickle.load(open(path + "raparams.pkl", "rb"))
        # m_list = pickle.load(open(path + "/m.pkl", "rb"))
        # v_list = pickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            # optimizer.m[name].set_value(m_list[name].astype(theano.config.floatX))
            # optimizer.v[name].set_value(v_list[name].astype(theano.config.floatX))
        for name in ra_list.keys():
            self.raparams[name].set_value(ra_list[name].astype(theano.config.floatX))


   