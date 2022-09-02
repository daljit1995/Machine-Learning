import cv2

# This function use for read and show the file
# How to read image use this sytax
img=cv2.imread(r'E:\Coding Notes\test\with_mask\1-with-mask.jpg')


cv2.imshow('Frame',img)
# When we put the 0 value in waitkey then image stable time infinit but when we put any value then close as per define time
cv2.waitKey(0)
# It use for close the window
cv2.destroyAllWindows()


# this function use for the Video on
import cv2
cap=cv2.VideoCapture(0) # It is use for open webacm 
while(cap.isOpened()): # it is use for open the camara
    _,img=cap.read() # Use _ this for skip boolean value in this function
    #img=img*2 # it is use for increase brithness
    cv2.imshow('Frame',img)
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Part 1. - Data Preprocessing

# We set the editing parameters in train


train_gen=ImageDataGenerator(rescale=1/255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)
test_gen=ImageDataGenerator(rescale=1/255)

train_set=train_gen.flow_from_directory(r'E:/Coding Notes/train',target_size=(150,150),batch_size=16,class_mode='binary')
test_set=train_gen.flow_from_directory(r'E:/Coding Notes/test',target_size=(150,150),batch_size=16,class_mode='binary')


# CNN - Convolutional Neural Network (do Convolab on piture) - it is use to create the feature map
# CNN is automatically detect the dat from picture
# Edge( where chnage colour of picture instently)

# 1. Convolution (move outer side for data)

# 2. Pooling (reduce size) - it is method of reduction of data
# There is 2 parameter use in Pooling function
# 1. Filter
# 2. Stride

# 3. Send data to neural network
# Sobel Operator use for detecting the edge detector( feature Map )
# when we need to make feature map then we use the kernal function to create the matrix

# 1. Convolution
cnn=tf.keras.models.Sequential()

# This is convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=[150,150,3]))
# kernal_size - it is size of matrix

# 2. Pooling (In the pooling we reduce the feature map with help of pool size)
# There is two typ pooling
# 1. Max Pool
# 2.Average Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# pool Size 2 means  2 by 2 matrix
# Strides (it is move to skip the row as per strides which is put in pool size)
 

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Flatten Layer
# Flatten - It is use for  converting any dimesion to 1 dimension

cnn.add(tf.keras.layers.Flatten())

# ANN
# Structure
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) 
# Sigmoid use in last layer for binary class
# Softmax Function 
# Compiler
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) # 'precision'

# Call backs - 
callbacks=[tf.keras.callbacks.EarlyStopping()]


# Fitt
cnn.fit(train_set,validation_data=test_set,epochs=100,callbacks=callbacks)


# To handle the Overfitting use dropout layer


