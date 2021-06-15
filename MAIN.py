#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tensorflow.python.framework import ops
import scipy.io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import interpolation
from tensorflow.keras.constraints import MinMaxNorm,NonNeg

from keras import layers

import tensorflow as tf
from skimage.metrics import structural_similarity as ssim


def save_results(Path,results_metrics,Best_metric,rho,learning_rate,rank,regularization_abund,Abun_layer,layer_num):
  if path.exists(Path + '/Results_summary.txt'):
    f = open(Path + '/Results_summary.txt', "a+")
  else:
    f = open(Path + '/Results_summary.txt', "w+")
    f.write('PSNR_spatial_LAST = ' + (str(results_metrics[0])[0:6]))
    f.write(', PSNR_spectral_LAST = ' + (str(results_metrics[1])[0:6]))
    f.write(', PSNR_gen_LAST = ' + (str(results_metrics[2])[0:6]))
    f.write(', RMSE_lAST = ' + (str(results_metrics[3])[0:6]))
    f.write(', SNR_LAST = ' + (str(results_metrics[4])[0:6]))
    f.write(', SAM_LAST = ' + (str(results_metrics[5])[0:6]))
    f.write(', ssim_const_LAST = ' + (str(results_metrics[6])[0:6]))
    f.write(', PSNR_spatial_BEST = ' + (str(Best_metric[0])[0:6]))
    f.write(', PSNR_spectral_BEST = ' + (str(Best_metric[1])[0:6]))
    f.write(', PSNR_gen_BEST = ' + (str(Best_metric[2])[0:6]))
    f.write(', RMSE_BEST = ' + (str(Best_metric[3])[0:6]))
    f.write(', SNR_BEST = ' + (str(Best_metric[4])[0:6]))
    f.write(', SAM_BEST = ' + (str(Best_metric[5])[0:6]))
    f.write(', ssim_const_BEST = ' + (str(Best_metric[6])[0:6]))
    f.write(', Abun_layer = ' + (str(Abun_layer)))
    f.write(', layer_num = ' + (str(layer_num)))
    f.write(', rho= '+ str(rho) +
              ', learning_rate= '+ str(learning_rate)+
              ', rank = '+ str(rank) + ', regularization_abund = ' + str(regularization_abund)+ "\n")
    
  return None

def Add_weights(x,weights):
    Aux = weights*x[0] + (1-weights)*x[1]
    return Aux

def Abundance_loss():
  def lossimage(y_true, y_pred):
      # batch x W x H X r
    val = tf.reduce_sum(tf.square(y_true - tf.reduce_sum(y_pred,axis=3)))
    return val
  return lossimage

class Endmemebers_Layer(Layer):
    def __init__(self, L=16, rank=1, initicial=None, **kwargs):
        self.L = L
        self.rank = rank
        if initicial is not None:
            self.initicial = np.expand_dims(np.expand_dims(np.expand_dims(initicial, 0), 0), 0)
        else:
            self.initicial = np.random.uniform(0, 1, (1, 1, 1, self.rank, self.L))

        super(Endmemebers_Layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'L': self.L,
            'rank': self.rank})
        return config

    def build(self, input_shape):

        Endmemb = tf.constant_initializer(self.initicial)
        self.Endmemb = self.add_weight(name='Endmemb', shape=(1, 1, 1, self.rank, self.L), initializer=Endmemb,
                                       constraint=NonNeg(), trainable=True)

    def call(self, inputs, **kwargs):
        # Hy = Rank_operator(inputs,self.Endmemb)
        Ab = tf.expand_dims(inputs, -1)
        Hy = tf.reduce_sum(tf.multiply(self.Endmemb, Ab), axis=3)
        return Hy

def performance(img,estimate):

  [M,N,L] = img.shape

  def fun_PSNR_2D(img,res):
    [M,N]=img.shape
    temp=1./(M*N)*np.sum(np.power(img-res,2))
    psnr= 10*np.log10(np.max(np.power(img,2)/temp))
    return psnr

  def mean_SAM (img,estimate):
    pp = np.expand_dims(np.sqrt(np.sum(np.power(img,2),2)),-1)
    pp2 = np.expand_dims(np.sqrt(np.sum(np.power(estimate,2),2)),-1)

    img = np.divide(img,pp)
    estimate = np.divide(estimate,pp2)

    z = np.sum(np.multiply(img,estimate),2)
    z = np.mean(np.arccos(z-2.2e-16)*180/np.pi)
    return z

  def fun_PSNR_1D(img,res):
    [M]=img.shape
    temp=1./(M)*np.sum(np.power(img-res,2))
    psnr= 10*np.log10(np.max(np.power(img,2)/temp))
    return psnr

  def fun_PSNR_3D(img,res):
    [M,N,L]=img.shape
    temp=1./(M*N*L)*np.sum(np.power(img-res,2))
    psnr= 10*np.log10(np.max(np.power(img,2)/temp))
    return psnr


  PSNR_spatial = 0
  for i in range(L):
    PSNR_spatial += fun_PSNR_2D(img[:,:,i],estimate[:,:,i])
  PSNR_spatial = PSNR_spatial/L

  PSNR_spectral =0
  for i in range(M):
    for j in range(N):
      PSNR_spectral += fun_PSNR_1D(img[i,j,:],estimate[i,j,:])
  PSNR_spectral = PSNR_spectral/(M*N)

  PSNR_gen = fun_PSNR_3D(img,estimate)

  RMSE = np.sqrt(np.mean(np.power(img-estimate,2)));

  SNR=10*np.log10(np.mean(np.power(img,2))/np.mean(np.power(estimate-img,2)));
  
  SAM = mean_SAM(img,estimate)

  ssim_const = ssim(img, estimate,
                  multichannel=True, win_size=3)
  
  
  


  return PSNR_spatial,PSNR_spectral,PSNR_gen,RMSE,SNR,SAM,ssim_const


def Abund_conv_net(inputs,rank,name='',L=16,number_layer=4):
    conv1 = Conv2D(L, 3, activation='relu', padding='same')(inputs)
    for layer_v in range(number_layer):
        conv1 = Conv2D(L, 3, activation='relu', padding='same')(conv1)
    final = Conv2D(rank, 1, activation='softmax', padding='same', name='Abund_final'+str(name))(conv1)

    return final

def Abund_net_autoencoder(inputs,rank,name='',L=16,number_layer=4):

    decrese = np.floor(number_layer/2)
    conv1 = Conv2D(L, 3, activation='relu', padding='same')(inputs)
    for layer_v in range(number_layer-1):
        if (layer_v<decrese):
          conv1 = Conv2D(np.int((layer_v+2)*L), 3, activation='relu', padding='same')(conv1)
        else:
          conv1 = Conv2D(np.int((decrese*2-layer_v)*L), 3, activation='relu', padding='same')(conv1)
    final = Conv2D(rank, 1, activation='softmax', padding='same', name='Abund_final'+str(name))(conv1)

    return final

def Abund_net_residual(inputs,rank,name='',L=16,number_layer=4):
    conv1 = Conv2D(L, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(L, 3, activation='relu', padding='same')(conv1)
    for layer_v in range(number_layer-2):
        conv2 = Conv2D(L, 3, activation='relu', padding='same')(conv2)
    conv_concat = concatenate([conv1,conv2], axis=3)
    final = Conv2D(rank, 1, activation='softmax', padding='same', name='Abund_final'+str(name))(conv_concat)

    return final


def VisualGraphs(RstActual,RstBest,ZTuckerRepr,ConvCurve,Original,ColorBands):
    clear_output(wait=True)
    
    [m,n,L]=Original.shape
    
    #Choose 3 uniform bands from low-Rank Tucker Representation of Z            
    bands = np.floor( np.linspace(L/4, 3*L/4, num=3)).astype(int)
    xo = ZTuckerRepr[:,:,[bands[0],bands[1],bands[2]]]
    xo[:,:,0] = xo[:,:,0]/np.max(xo[:,:,0])
    xo[:,:,1] = xo[:,:,1]/np.max(xo[:,:,1])
    xo[:,:,2] = xo[:,:,2]/np.max(xo[:,:,2])
    
    ErrActual = np.divide(np.power(np.sum(np.power(RstActual-Original,2),axis=2),0.5),np.power(np.sum(np.power(Original,2),axis=2),0.5))
    
    ErrBest = np.divide(np.power(np.sum(np.power(RstBest-Original,2),axis=2),0.5),np.power(np.sum(np.power(Original,2),axis=2),0.5))
    
    fig, axs = plt.subplots(1,4,figsize=(14,14))
    fig.subplots_adjust(left=.05, bottom=0.1, right=.9, top=0.9, wspace=0.05)            
    
    RGB = Original[:,:,ColorBands]/np.max(Original)
    RGB = RGBZoom(RGB)
    axs[0].imshow(RGB)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    RGB = RstBest[:,:,ColorBands]/np.max(RstBest)
    RGB = RGBZoom(RGB)
    axs[1].imshow(RGB)
    axs[1].set_title('Best Reconstruction')
    axs[1].axis('off')
    
    im = axs[2].imshow(ErrBest, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[2].set_title('Relative Error Map\n Best Reconstruction')
    axs[2].axis('off')
    
    axs[3].imshow(xo)
    axs[3].set_title('3Bands from Tucker \n Representation of Z')
    axs[3].axis('off')
    
    fig, axs = plt.subplots(1,4,figsize=(14,14))
    fig.subplots_adjust(left=.05, bottom=0.1, right=.9, top=0.9, wspace=0.05)            
    
    asp = np.diff(axs[0].get_xlim())[0] / np.diff(axs[0].get_ylim())[0]
    axs[0].set_aspect(asp)
    axs[0].axis('off')
    
    RGB = RstActual[:,:,ColorBands]/np.max(RstBest)
    RGB = RGBZoom(RGB)
    axs[1].imshow(RGB)
    axs[1].set_title('Actual Reconstruction')
    axs[1].axis('off')
    
    im = axs[2].imshow(ErrActual, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[2].set_title('Relative Error Map\n Actual Reconstruction')
    axs[2].axis('off')
    
    axs[3].plot(ConvCurve)
    axs[3].set_title('Convergence Curve\n (Best/Actual)PSNR: (%.2f,' %np.max(ConvCurve) +'%.2f)'%ConvCurve[-1])
    asp = np.diff(axs[3].get_xlim())[0] / np.diff(axs[3].get_ylim())[0]
    axs[3].set_aspect(asp)
    axs[3].yaxis.tick_right()
    
    plt.show()

def addGaussianNoise(y,SNR):
    
    sigma = np.sum(np.power(y,2))/(np.product(y.shape)*10**(SNR/10));
    w = np.random.normal(0, np.sqrt(sigma),size =y.shape);
    return y+w;


    return y+noise_y

def RGBZoom(X):
    
    w = 6
    y = [6, 100+6]
    x = [6, 100+6]
    
    for i in range(3):
        Y = interpolation.zoom(X[75:107,101:133,i],100/32)   
        X[y[0]:y[1],x[0]:x[1],i] = Y
    
    
    X[y[0]-w:y[0],x[0]-w:x[1]+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]:y[1]+1+w,x[0]-w:x[0],0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    w = 2
    y = [74, 107]
    x = [100, 133]    
    X[y[0]-w:y[0],x[0]-w:x[1]+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]:y[1]+1+w,x[0]-w:x[0],0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
            
            
    return X

    
def Hxfunction(x,largo,ancho,profun,H):
    Aux = tf.reshape(x,(largo,ancho,profun))
    Aux = tf.transpose(Aux,perm=[2,1,0])
    Aux = tf.reshape(Aux,(largo*ancho*profun,1))
    
    #print(Aux.shape)
    
    Aux = tf.sparse.sparse_dense_matmul(H,tf.cast(Aux,dtype=tf.float64))
    Aux = tf.reshape(Aux,(1,1,1,H.shape[0]))
    return Aux





class XoLayer(layers.Layer):
    def __init__(self, largo=256, ancho=256, profun=10, fact=0.3):
        super(XoLayer, self).__init__()

        self.largo = largo
        self.ancho = ancho
        self.profun = profun

        # self.largo_fac = tf.cast(tf.math.round(largo*fact),dtype=tf.int32)
        # self.ancho_fac = tf.cast(tf.math.round(ancho*fact),dtype=tf.int32)
        # self.profun_fac = tf.cast(tf.math.round(profun*fact),dtype=tf.int32)

        self.largo_fac = int(np.round(largo * fact))
        self.ancho_fac = int(np.round(ancho * fact))
        self.profun_fac = int(np.round(profun * fact))

        # self.kernel = self.add_weight(shape=(tf.cast(tf.math.round(profun*fact),dtype=tf.int32),tf.cast(tf.math.round(largo*fact),dtype=tf.int32)*tf.cast(tf.math.round(ancho*fact),dtype=tf.int32)),
        #                     initializer='glorot_normal',#'glorot_normal',
        #                     trainable=True)

        self.kernel = self.add_weight(shape=(self.profun_fac, self.largo_fac * self.ancho_fac),
                                      initializer='glorot_normal',  # 'glorot_normal',
                                      trainable=True)

        self.Dx = self.add_weight(shape=(largo, self.largo_fac),
                                  initializer='uniform',  # uniform
                                  trainable=True)
        self.Dy = self.add_weight(shape=(ancho, self.ancho_fac),
                                  initializer='uniform',
                                  trainable=True)
        self.Dz = self.add_weight(shape=(profun, self.profun_fac),
                                  initializer='uniform',
                                  trainable=True)

    def call(self, inputs):
        Aux = tf.transpose(tf.matmul(self.Dz, self.kernel))
        Aux = tf.reshape(Aux, (self.largo_fac, self.ancho_fac * self.profun))

        Aux = tf.matmul(self.Dx, Aux)
        Aux = tf.reshape(Aux, (self.largo, self.ancho_fac, self.profun))
        Aux = tf.transpose(Aux, perm=[1, 0, 2])
        Aux = tf.reshape(Aux, (self.ancho_fac, self.ancho * self.profun))
        Aux = tf.matmul(self.Dy, Aux)
        Aux = tf.reshape(Aux, (self.ancho, self.largo, self.profun))
        Aux = tf.reshape(tf.transpose(Aux, perm=[1, 0, 2]), (1, self.ancho, self.largo, self.profun))

        return Aux


        
    def call(self, inputs):
        
        Aux = tf.transpose(tf.matmul(self.Dz,self.kernel))
        Aux = tf.reshape(Aux,( self.largo_fac,self.ancho_fac*self.profun))

        Aux = tf.matmul(self.Dx,Aux)
        Aux = tf.reshape(Aux,(self.largo,self.ancho_fac,self.profun))
        Aux = tf.transpose(Aux,perm=[1,0,2])
        Aux = tf.reshape(Aux,(self.ancho_fac,self.ancho*self.profun))
        Aux = tf.matmul(self.Dy,Aux)
        Aux = tf.reshape(Aux,(self.ancho,self.largo,self.profun))
        Aux = tf.reshape(tf.transpose(Aux,perm=[1,0,2]),(1,self.ancho,self.largo,self.profun))
        
        return  Aux
    

def fun_PSNR(img,res):

    [M,N,L]=img.shape
    temp=1./(M*N*L)*np.sum(np.power(img-res,2))
    psnr= 10*np.log10(np.max(np.power(img,2)/temp))
    return psnr    

def psnr(img, res):
  img = np.squeeze(img)
  res = np.squeeze(res)
  [M, N, L] = img.shape
  temp = 1. / (M * N * L) * np.sum(np.power(img - res, 2))
  psnr = 10 * np.log10(np.max(np.power(img, 2) / temp))
  return psnr


def residualNet(pretrained_weights = None,input_size = (256,256,1), L=10, H=0, fact = 0.5):
    
    inputs = Input(input_size)
    inicial = XoLayer(largo = input_size[0], ancho = input_size[1], profun = L, fact = fact)(inputs)
    
    drop1 = Dropout(0.2)(inicial)
    conv1 = Conv2D(L,3,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(drop1)
    conv1 = Conv2D(L,1,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(conv1)
    
    
    conv8 = Conv2D(L,3,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(conv1)
    conv8 = Conv2D(L,1,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(conv8)
    
    
    conv8 = Add()([conv8,conv1])
    
    final =Lambda(lambda x: Hxfunction(x,largo=input_size[0],ancho=input_size[1],profun=L,H=H)) (conv8)
    
    model = Model(input = inputs, output = final)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def Unmixing_net(pretrained_weights=None, input_size=(512, 512, 31), batch_size=10):
    M = input_size[0]
    N = input_size[1]
    L = input_size[2]
    inputs = Input(input_size)
    # inputs = tf.reshape(inputs,[M*N,L])
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(16, 3, activation='relu', padding='same')(conv3)

    abund = Conv2D(4, 3, activation='relu', padding='same', name='abund')(conv4)
    # batch x W x H X r
    # batch x 1 x L x MN
    # custim layer que haga la reorgani()zation

    inputs2 = AveragePooling2D(pool_size=(4, 4), name='H2_P2')(inputs)
    # reorg = Lambda(lambda x: tf.reshape(tf.expand_dims(tf.reshape(x,[batch_size,int(M/4)*int(N/4),L]),axis=1)))(inputs2)
    reorg = Lambda(
        lambda x: tf.transpose(tf.expand_dims(tf.reshape(x, [batch_size, int(M / 4) * int(M / 4), L]), axis=1),
                               [0, 1, 3, 2]))(inputs2)

    conv5 = Conv2D(32 * 32, [1, 10], activation='relu', padding='same')(reorg)
    conv6 = Conv2D(16 * 16, [1, 10], activation='relu', padding='same')(conv5)
    conv7 = Conv2D(8 * 8, [1, 10], activation='relu', padding='same')(conv6)
    conv8 = Conv2D(4 * 4, [1, 10], activation='relu', padding='same')(conv7)
    conv9 = Conv2D(2 * 2, [1, 10], activation='sigmoid', padding='same')(conv8)

    endme = Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]), name='endme')(conv9)

    final_capa = Lambda(lambda x: tf.matmul(x[0], x[1]), name='final')((abund, endme))

    model = Model(inputs, [final_capa, abund])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def UNetL(pretrained_weights = None,input_size = (256,256,1), L=10, H=0, fact = 0.5):
    L_2 = 2*L;
    L_3 = 3*L;
    L_4 = 4*L;
    
    inputs = Input(input_size)  
    inicial = XoLayer(largo=input_size[0], ancho=input_size[1], profun=L, fact=fact)(inputs)
    
    drop1 = Dropout(0.2)(inputs)

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop1)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)    

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal',)(conv7)

    conv8 = Conv2D(L, 1)(conv7)
    
    final =Lambda(lambda x: Hxfunction(x,largo=input_size[0],ancho=input_size[1],profun=L,H=H)) (conv8)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def Auto_encoder(pretrained_weights=None, input_size=(256, 256, 1), L=10, H=0, fact=0.5):
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    
    inputs = Input(input_size)    
    inicial = XoLayer(largo=input_size[0], ancho=input_size[1], profun=L, fact=fact)(inputs)
    
    # the encoder part
    conv1 = Dropout(0.2)(inicial)    
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4))(conv3)
    
    convup2=Conv2DTranspose(filters=L_3, kernel_size=(3, 3), strides=4, activation='relu', padding='same')(pool3)
    convup3=Conv2DTranspose(filters=L_2, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(pool2)
    conv8=Conv2DTranspose(filters=L, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(convup3)


    final = Lambda(lambda x: Hxfunction(x, largo=input_size[0],ancho=input_size[1],profun=L, H=H))(conv8)

    model = Model(input=inputs, output=final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



from keras import backend as K
class myCallback(tf.keras.callbacks.Callback):
     

    def __init__(self,Xorig=0,Freq=0):
        super(myCallback, self).__init__()
        self.my_PSNR = []
        self.Xorig = Xorig;
        self.Best = np.zeros(shape=Xorig.shape);
        self.Freq = Freq
        

    def on_epoch_end(self, epoch, logs={}):
        Freq = self.Freq
        self.model.layers[2].rate=0.0
        
        
        if np.mod(epoch,Freq)==0:            
            img = self.Xorig;
            
            
            [m,n,L] = img.shape
           
            #func = K.function([self.model.layers[0].input],[self.model.layers[0].output])
            #xo = func(np.zeros(shape=(m,n,L)))
            #xo = np.asarray(xo).reshape((m,n,L),order="F")
            
            func = K.function([self.model.layers[0].input],[self.model.layers[len(self.model.layers)-2].output])
            result = func(np.zeros(shape=(1,m,n,L)))
            result = np.asarray(result).reshape((m,n,L),order="F")
            
            
            psnr = fun_PSNR(img,result)
            self.my_PSNR.append(psnr) 
            print('Epoch %05d: PSNR %6.3f : Max PSNR %6.3f' % (epoch, psnr,np.max(self.my_PSNR)))
            
            if psnr >= np.max(self.my_PSNR):                
                self.Best = result
                setattr(self.model, 'Best', self.Best)
            
                                
            setattr(self.model, 'PSNRs', self.my_PSNR)
            
        if np.mod(epoch,Freq*5)==0: 

            self.model.layers[2].rate=0.5
            
            VisualGraphs(result,self.Best,self.Xorig,self.my_PSNR,img,[8,4,2])
           