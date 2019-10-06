#python2 funciones_basicas.py
import numpy as np
from math import *
import cv2

def translation(image,tx,ty):
    rows,cols,channels=image.shape
    new_image=np.zeros((rows+tx,cols+ty,channels),np.uint8)
    for i in range(0,rows):
        for j in range(0,cols):
            new_image[i+tx][j+ty]=image[i][j]
    return new_image

def to_radians(angle):
    return angle*pi/180

def rotation(image,angle):
    angle=to_radians(angle)
    rows,cols,channels=image.shape  
    
    new_image=np.zeros((rows,cols,channels),np.uint8)
    new_image = cv2.resize(new_image, (1000,1000))
    center_x=rows//2
    center_y=cols//2
    
    for i in range(0,rows):
        for j in range(0,cols):
            origen_x=j-center_x
            origen_y=i-center_y
            ty=(origen_x*cos(angle)+origen_y*sin(angle))+abs(int(cols*sin(angle)))
            tx=(-origen_x*sin(angle)+origen_y*cos(angle))+abs(int(100+sin(angle)))

            tx=tx+center_x
            ty=ty+center_y
            #if(0<=ty<rows and 0<=tx<cols):
            new_image[int(tx)][int(ty)]=image[i][j]

    return new_image

def scaling(image, scale):
    rows,cols,channels=image.shape
    new_image=np.zeros((rows*scale,cols*scale,channels),np.uint8)

    for i in range(0,rows):
        for j in range(0,cols):
            tx=i*scale
            ty=j*scale
            cont=0
            '''while(scale>1){
                new_image[int(tx+cont)][int(ty+cont)]=image[i][j]
                cont+=1
            }'''
            new_image[int(tx)][int(ty)]=image[i][j]
            
    return new_image

image=cv2.imread('lena.jpg',1)
new_image=image
#cv2.imshow('Imagen original',image)

#new_image=translation(image,80,100)
new_image=scaling(image,1.5)
#new_image=rotation(image,180)

cv2.imshow('Transformacion geometrica',new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()