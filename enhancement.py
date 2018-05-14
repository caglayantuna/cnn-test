#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

import sys, getopt, gdal
from osgeo.gdalconst import GA_ReadOnly
import scipy.misc as scipy
import numpy as np
import glob, os, re
import tensorflow as tf
from MODEL import model
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32


scale=4
def Ihs_Forward(im):
    I=im[0,:,:]/3+im[1,:,:]/3+im[2,:,:]/3
    v1=((-1/np.sqrt(6))*im[0,:,:])+((-1/np.sqrt(6))*im[1,:,:])+((2/np.sqrt(6)))*im[2,:,:]
    v2=((1/np.sqrt(6))*im[0,:,:])+((-1/np.sqrt(6))*im[1,:,:])
    return I,v1,v2
def Ihs_Reverse(I,v1,v2,rows,cols):
    result=np.zeros([3,scale*rows,scale*cols])
    result[0,:,:]=I+(-1/np.sqrt(6))*v1+(3/np.sqrt(6))*v2;
    result[1,:,:]=I+(-1/np.sqrt(6))*v1+(-3/np.sqrt(6))*v2;
    result[2,:,:]=I+((2/np.sqrt(6))*v1);
    return result
def image_reshape(image):
    
    image_band1=image[0,:,:]
    image_band2=image[1,:,:]
    image_band3=image[2,:,:]
    image_allband = np.zeros((scale*rows,scale*cols,3),dtype='uint16') 
    image_allband[:,:,0]=image_band1
    image_allband[:,:,1]=image_band2
    image_allband[:,:,2]=image_band3
    return image_allband
gdal.AllRegister()

Data = gdal.Open('itu_phr_cut_5.tif',GA_ReadOnly) 
cols =  Data.RasterXSize
rows =  Data.RasterYSize    
bands = Data.RasterCount  

image=np.array(Data.ReadAsArray())
imf=image.astype(float)
imres=np.zeros([4,scale*rows,scale*cols])
imres[0,:,:]=scipy.imresize(imf[0,:,:], [scale*rows,scale*cols],interp='bicubic',mode='F')
imres[1,:,:]=scipy.imresize(imf[1,:,:], [scale*rows,scale*cols],interp='bicubic',mode='F')
imres[2,:,:]=scipy.imresize(imf[2,:,:], [scale*rows,scale*cols],interp='bicubic',mode='F')
imres[3,:,:]=scipy.imresize(imf[3,:,:], [scale*rows,scale*cols],interp='bicubic',mode='F')

I,v1,v2=Ihs_Forward(imres)    

norma=I/np.amax(I)

model_list = sorted(glob.glob("./VDSR_adam_epoch_*"))
model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
model_ckpt=model_list[0]

with tf.Session() as sess:
   	
   input_tensor  			= tf.placeholder(tf.float32, shape=(1, None, None, 1))
   shared_model = tf.make_template('shared_model', model)
   output_tensor, weights 	= shared_model(input_tensor)

   saver = tf.train.Saver(weights)
   saver.restore(sess, model_ckpt)
   img_vdsr= sess.run([output_tensor], feed_dict={input_tensor: np.resize(norma, (1, norma.shape[0], norma.shape[1], 1))})

img_vdsr = np.resize(img_vdsr, (norma.shape[0], norma.shape[1]))

img_vdsr=img_vdsr*np.amax(I)
imn=Ihs_Reverse(img_vdsr,v1,v2,rows,cols)
imn=image_reshape(imn)
driver = gdal.GetDriverByName('GTiff')
outDataset = driver.Create('itu_vdsr_4.tiff', scale*cols,scale*rows,3,gdal.GDT_UInt16)
projection = Data.GetProjection()
geotransform = Data.GetGeoTransform()
gt=list(geotransform)
gt[1]=0.125
gt[5]=-0.125
outDataset.SetProjection(projection)        
outDataset.SetGeoTransform(tuple(gt))
for k in range(3):        
 outBand = outDataset.GetRasterBand(k+1)
 outBand.WriteArray(imn[:,:,k]) 
 outBand.FlushCache() 
      


