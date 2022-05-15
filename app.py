import streamlit as st
import numpy as np
import tensorflow as tf
import time
import os
import sys

from ops import batch_norm,linear,conv2d,deconv2d,lrelu
from matplotlib import pyplot as plt
from image_helpers import *
from PIL import Image
from io import BytesIO
import base64

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}

def main():
  st.title("Face Completion")	
  st.subheader("Streamlit From Colab")	
  is_crop=True
  batch_size=64
  image_size=108
  sample_size=64
  image_shape=[64,64,3]

  z_dim=100

  gf_dim=64
  df_dim=64

  learning_rate=0.0002
  beta1=0.5
  batch_z=np.random.uniform(-1,1,[batch_size,z_dim]).astype(np.float32)
  #mask
  scale=0.25
  mask_=np.ones([batch_size]+image_shape).astype(np.float32)
  l=int(64*scale)
  u=int(64*(1.0-scale))
  mask_[:,l:u,l:u,:]=0.0
  #inverse mask
  scale=0.25
  imask_=np.zeros([batch_size]+image_shape).astype(np.float32)
  l=int(64*scale)
  u=int(64*(1.0-scale))
  imask_[:,l:u,l:u,:]=1.0
  match_file = st.file_uploader(
                      label="Enter the photo to be matched", type=['jpg'])
  
  if match_file is not None:
    data = Image.open(match_file)
    data1= np.array(data)
    
    
    sample_z=np.random.uniform(-1,1,size=(sample_size,z_dim))
    
    
    sample=get_image(data1, image_size,is_crop)
    st.write('sample')
    st.write(sample.shape)

    
    INPUT = []
    x = np.zeros((64,64,3))
    INPUT.append(sample)

    for i in range(63):
      INPUT.append(x)

    INPUT = np.array(INPUT)
    st.write('uploaded image')
    st.image(INPUT[0], clamp=True, channels='RGB')
    st.write(INPUT.shape)
    st.write('loading model')

    with tf.compat.v1.Session() as sess:
      graph = tf.get_default_graph()
      saver = tf.compat.v1.train.import_meta_graph('model/Model.cpkt.meta')
      saver.restore(sess,tf.train.latest_checkpoint('model/'))
      
      images=graph.get_tensor_by_name('real_images:0')
      z=graph.get_tensor_by_name("z:0")
      g=graph.get_tensor_by_name("Gen:0")

      sample_generated=sess.run([g],feed_dict={z:sample_z,images:INPUT})
      original_part=np.multiply(INPUT,mask_)
      generated_part=np.multiply(sample_generated,imask_)
      total=np.add(original_part,generated_part)
      
      st.write('generated completed face')
      st.image(np.array(total[0][0]), clamp=True, channels='RGB' )
      

if __name__ == '__main__':
	main()
