import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pycocotools import coco
import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import copy
import os
from pprint import pprint
from pycocotools.coco import COCO
import skimage.io as io
import pylab
from PIL import Image
from resizeimage import resizeimage

ids=[]
anns=[]
label=[]
text_freq=[]
X_=[]
Y_=[]
with open('COCO_Text.json') as json_file:
    coco_text=json.load(json_file)
    print ('Loaded')
    for r in coco_text["imgs"]:
        ids.append(r)
    for j in coco_text["anns"]:
        anns.append(j)
    for x in coco_text["imgToAnns"]:
        label.append(x)
    for k in label:
        text_freq.append(coco_text["imgToAnns"][k])

img_names=[]
for a in range (10000):
    img_names.append(coco_text["imgs"][ids[a]]["file_name"])

for i in range (len(img_names)):
    I=Image.open('./trainingset1/%s'%(img_names[i]))
    I=np.array(I)
    I=I.ravel()
    X_.append(I)

for h in range(len(text_freq)):
    if len(text_freq[h])==0:
        text_freq[h]=0
    else:
        text_freq[h]=1


for x in range(10000):
    Y_.append(text_freq[x])


n_hu_l1 = 500
n_hu_l2 = 500
n_hu_l3 = 500
batch_size=100
hm_epochs=10



del ids, anns, label, text_freq


x_train=X_[0:6000]
x_val=X_[6000:8000]
x_test=X_[8000:10000]
y_train=Y_[0:6000]
y_val=Y_[6000:8000]
y_test=Y_[8000:10000]
x_proba=X_[0:1000]
y_proba=Y_[0:1000]

sess = tf.InteractiveSession()


x = tf.placeholder("float", [None,187500])
y_ = tf.placeholder("float")


W = tf.Variable(tf.random_uniform([187500, n_hu_l1], minval=0,maxval=1))
b = tf.Variable(tf.random_uniform([n_hu_l1], minval=0,maxval=1))
hidden  = tf.nn.relu(tf.matmul(x,W) + b) # first layer.

W2 = tf.Variable(tf.random_uniform([n_hu_l1,n_hu_l2],minval=0,maxval=1))
b2 = tf.Variable(tf.random_uniform([n_hu_l2], minval=0,maxval=1))
hidden2 = tf.nn.relu(tf.matmul(hidden, W2)+b2)

W3 = tf.Variable(tf.random_uniform([n_hu_l2,n_hu_l3],minval=0,maxval=1))
b3 = tf.Variable(tf.random_uniform([n_hu_l2], minval=0,maxval=1))
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3)+b3)

W4 = tf.Variable(tf.random_uniform([n_hu_l3,1],minval=0,maxval=1))
b4 = tf.Variable(tf.random_uniform([1], minval=0,maxval=1))
hidden4 = tf.matmul(hidden3, W4)


y = tf.nn.softmax(hidden4)



cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)


tf.initialize_all_variables().run()
for epoch in range (hm_epochs):
    for _ in range(int(len(x_proba)/batch_size)):
        ind=0
        epoch_x=x_proba[ind:(batch_size-1)][:]
        epoch_y=y_proba[ind:(batch_size-1)][:]
        feed_dict={x: epoch_x, y_:epoch_y }
        e,a=sess.run([cross_entropy,train_step],feed_dict)
        ind+=100
        if e<1:break # early stopping yay
print "step %d : entropy %s" % (epoch,e)




correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_proba,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "accuracy %s"%(accuracy.eval({x: x_proba, y_: y_proba}))

learned_output=tf.argmax(y,1)
print learned_output.eval({x: x_proba})
