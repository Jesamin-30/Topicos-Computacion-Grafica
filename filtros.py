#python2 filtros.py
import numpy as np
from math import *
import cv2


#G=[[1,2,1],[2,4,2],[1,2,1]] #1/16
G=[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]] #1/246

def media(img, mask=5):
    rows,cols=image.shape
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    center=int(mask/2)
    for i in range(center,rows-center):
        for j in range(center, cols-center):
            cont=0
            for k in range(i-center,i+center+1):
                for l in range(j-center,j+center+1):
                    cont=cont+image[k][l]
            new_image[i][j]=cont/(mask*mask)
            
    cv2.imshow('Media',new_image) 

def median(img, mask=7):
    rows,cols=image.shape
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    center=int(mask/2)
    for i in range(center,rows-center):
        for j in range(center, cols-center):
            list=[]
            for k in range(i-center,i+center+1):
                for l in range(j-center,j+center+1):
                    list.append(image[k][l])
            list.sort()
            new_image[i][j]=list[int((mask*mask)/2)]
            
    cv2.imshow('Mediana',new_image) 

def max(img, mask=3):
    rows,cols=image.shape
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    center=int(mask/2)
    for i in range(center,rows-center):
        for j in range(center, cols-center):
            list=[]
            for k in range(i-center,i+center+1):
                for l in range(j-center,j+center+1):
                    list.append(image[k][l])
            list.sort()
            new_image[i][j]=list[(mask*mask)-1]
            
    cv2.imshow('Max',new_image) 

def min(img, mask=3):
    rows,cols=image.shape
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    center=int(mask/2)
    for i in range(center,rows-center):
        for j in range(center, cols-center):
            list=[]
            for k in range(i-center,i+center+1):
                for l in range(j-center,j+center+1):
                    list.append(image[k][l])
            list.sort()
            new_image[i][j]=list[0]
            
    cv2.imshow('Min',new_image) 

def gauss(image, mask=5):
    rows,cols=image.shape
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    center=int(mask/2)
    for i in range(0,rows):
        for j in range(0, cols):
            cont=0
            for k in range(0,mask):
                for l in range(0,mask):
                    if(i+k<rows and j+l<cols):
                        cont=cont+(image[i+k][j+l]*G[k][l])
            #new_image[i][j]=int(cont/16)            
            new_image[i][j]=int(cont/246)
    cv2.imshow('Gauss',new_image) 

def laplace(image, mask=3):
    rows,cols=image.shape   
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            cont= int(image[i][j+1])+int(image[i][j-1])+int(image[i+1][j])+int(image[i-1][j])-4*int(image[i][j])
            if(cont<0): 
                cont=0
            elif(cont>255): 
                cont=255
            new_image[i][j]=cont
    cv2.imshow('Laplace', new_image)

def roberts(img):
    rows,cols=image.shape   
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    Rx=image.copy()
    Ry=image.copy()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            cont_x=(-1)*int(image[i-1][j-1])+int(image[i][j])
            if(cont_x<0): 
                cont_x=0
            elif(cont_x>255): 
                cont_x=255
            Rx[i][j]=cont_x

            cont_y=(-1)*int(image[i-1][j])+int(image[i][j-1])
            if(cont_y<0): 
                cont_y=0
            elif(cont_y>255): 
                cont_y=255
            Ry[i][j]=cont_y
            
    new_image=Rx+Ry
    cv2.imshow('Roberts', new_image)

def sobel(img):
    rows,cols=image.shape   
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    Sx=image.copy()
    Sy=image.copy()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            cont_x=(-1)*int(image[i-1][j-1])+int(image[i-1][j+1])-2*int(image[i][j-1])+2*int(image[i][j+1])-int(image[i+1][j-1])+int(image[i+1][j+1])
            if(cont_x<0): 
                cont_x=0
            elif(cont_x>255): 
                cont_x=255
            Sx[i][j]=cont_x

            cont_y=(-1)*int(image[i-1][j-1])+int(image[i+1][j-1])-2*int(image[i-1][j])+2*int(image[i+1][j])-int(image[i-1][j+1])+int(image[i+1][j+1])
            if(cont_y<0): 
                cont_y=0
            elif(cont_y>255): 
                cont_y=255
            Sy[i][j]=cont_y
    new_image=Sx+Sy
    cv2.imshow('Sobel', new_image)
   
def prewitt(img):
    rows,cols=image.shape   
    new_image=image
    new_image=np.zeros((rows,cols),np.uint8)
    Px=image.copy()
    Py=image.copy()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            cont_x=(-1)*int(image[i-1][j-1])+int(image[i-1][j+1])-1*int(image[i][j-1])+1*int(image[i][j+1])-int(image[i+1][j-1])+int(image[i+1][j+1])
            if(cont_x<0): 
                cont_x=0
            elif(cont_x>255): 
                cont_x=255
            Px[i][j]=cont_x

            cont_y=(-1)*int(image[i-1][j-1])+int(image[i+1][j-1])-1*int(image[i-1][j])+1*int(image[i+1][j])-int(image[i-1][j+1])+int(image[i+1][j+1])
            if(cont_y<0): 
                cont_y=0
            elif(cont_y>255): 
                cont_y=255
            Py[i][j]=cont_y
    new_image=Px+Py
    cv2.imshow('Prewitt', new_image)

if __name__ == '__main__':
    image=cv2.imread('circuit.jpg',0)      

    #media(image)
    #median(image)
    #max(image)
    #min(image)
    #gauss(image)
    laplace(image)
    roberts(image)
    sobel(image)
    prewitt(image)
    cv2.imshow('Imagen original',image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       