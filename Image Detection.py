import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model=tf.keras.models.load_model(r'E:/Coding Notes/mymodel.h5')
cascade=cv2.CascadeClassifier(r'E:/Coding Notes/haarcascade_frontalface_default.xml')

# Haarcascade use for the face detection and it is train in c++ not in python

cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,frame=cap.read()
    faces=cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
    for x,y,w,h in faces:
        face=frame[y:y+h,x:x+w] # Select the size of image from faces
        cv2.imwrite('temp.jpg',face) # write or save the image
        face=image.load_img('temp.jpg',target_size=(150,150,3)) # load the image for detection
        face=image.img_to_array(face) # this function use for convert image into array form
        face=np.expand_dims(face,axis=0)
        ans=model.predict(face)
        if ans[0][0]<0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame,'With Mask',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(frame,'Without Mask',(20,20),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()