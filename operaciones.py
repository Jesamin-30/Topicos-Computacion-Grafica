#python2 operaciones.py
import numpy as np
from math import *
import cv2

def negative(value):
    return 255-value

def logarithmic(value):
    const=255/np.log2(1+255)    
    return const*np.log2(1+value)

def logarithmic_inverse(value):
    const=255*np.log2(256)    
    return const*(1/np.log2(1+value))

def power(value,gamma=0.5):
    c=1
    return c*pow(value,gamma)*255/(c*pow(255,gamma))

        
image=cv2.imread('woman.jpg',0)

#rows,cols,channels=image.shape
rows,cols=image.shape
#new_image=np.zeros((rows,cols,channels),np.uint8)
image_negative=np.zeros((rows,cols),np.uint8)
image_log=np.zeros((rows,cols),np.uint8)
image_power=np.zeros((rows,cols),np.uint8)
image_inverse=np.zeros((rows,cols),np.uint8)
for i in range(0,rows):
    for j in range(0,cols):
        image_negative[i][j]=negative(image[i][j])
        image_log[i][j]=logarithmic(image[i][j])
        image_power[i][j]=power(image[i][j])
        image_inverse[i][j]=logarithmic_inverse(image[i][j])

cv2.imshow('Imagen original',image)
cv2.imshow('Negative',image_negative)
cv2.imshow('Logarithmic',image_log)
cv2.imshow('Power',image_power)
cv2.imshow('Log Inverse',image_inverse)

cv2.waitKey(0)
cv2.destroyAllWindows()