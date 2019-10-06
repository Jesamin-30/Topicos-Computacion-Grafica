#python2 fourier.py
import numpy as np
from math import sqrt,exp
import cv2

#muestra imagen
def mlog(image):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    for i in range(0,rows):
        for j in range(0,cols):
            if(image[i][j]==0):
                new_image[i][j]=np.log(0.0000001)
            else:
                new_image[i][j]=np.log(image[i][j])
    return new_image
            
def normalize(image):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols),np.uint8)
    min=image.min()
    max=image.max()
    longitud=max-min

    if(longitud == 0):
        return new_image       

    for u in range(0,rows):
        for v in range(0,cols):
            new_image[u][v]=(image[u][v]-min)*255/longitud
    return new_image         

def shift(image):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols),image.dtype)
    m=rows//2
    n=cols//2
    new_image[0:m,0:n]=image[m:rows,n:cols]
    new_image[0:m,n:cols]=image[m:rows,0:n]
    new_image[m:rows,0:n]=image[0:m,n:cols]
    new_image[m:rows,n:cols]=image[0:m,0:n]
    return new_image
    

def fourier(image):
    rows,cols=image.shape

    x = np.arange(rows, dtype = float)
    y = np.arange(cols, dtype = float)

    u = x.reshape((rows,1))
    v = y.reshape((cols,1))

    exp_1 = pow(np.e, -2j*np.pi*u*x/rows)
    exp_2 = pow(np.e, -2j*np.pi*v*y/cols)
    dft=np.dot(exp_2, np.dot(exp_1,image).transpose())/(rows*cols)            
    
    return dft

def inverse_fourier(image,name):
    rows,cols=image.shape
    x = np.arange(rows, dtype = float)
    y = np.arange(cols, dtype = float)

    u = x.reshape((rows,1))
    v = y.reshape((cols,1))

    exp_1 = pow(np.e, 2j*np.pi*u*x/rows)
    exp_2 = pow(np.e, 2j*np.pi*v*y/cols)
    idft=np.dot(exp_2, np.dot(exp_1,image).transpose())

    x=np.abs(idft)
    
    cv2.imshow(name,normalize(x)) 

def ideal_low_pass_filter(image,d0):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    center_r=rows/2
    center_c=cols/2
    for u in range(0,rows):
        for v in range(0,cols):
            d=sqrt(pow(u-center_r,2)+pow(v-center_c,2))
            if(d<=d0):
                new_image[u][v]=1
            else:
                new_image[u][v]=0

    return shift(new_image)*image

def ideal_high_pass_filter(image,d0):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    center_r=rows/2
    center_c=cols/2
    for u in range(0,rows):
        for v in range(0,cols):
            d=sqrt(pow(u-center_r,2)+pow(v-center_c,2))
            if(d<=d0):
                new_image[u][v]=0
            else:
                new_image[u][v]=1

    return shift(new_image)*image

def butterworth_low_pass_filter(image,d0,n):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    center_r=rows/2
    center_c=cols/2
    for u in range(0,rows):
        for v in range(0,cols):
            d=sqrt(pow(u-center_r,2)+pow(v-center_c,2))
            new_image[u][v]=1/(1+pow(d/d0,2*n))

    return shift(new_image)*image

def butterworth_high_pass_filter(image,d0,n):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    center_r=rows/2
    center_c=cols/2
    for u in range(0,rows):
        for v in range(0,cols):
            d=sqrt(pow(u-center_r,2)+pow(v-center_c,2))
            if(d==0):
                d=0.0001
            new_image[u][v]=1/(1+pow(d0/d,2*n))

    return shift(new_image)*image

def gauss_low_pass_filter(image,d0):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    center_r=rows/2
    center_c=cols/2
    for u in range(0,rows):
        for v in range(0,cols):
            d=sqrt(pow(u-center_r,2)+pow(v-center_c,2))
            new_image[u][v]=exp(-(d*d)/(2*d0*d0))

    return shift(new_image)*image

def gauss_high_pass_filter(image,d0):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols))
    center_r=rows/2
    center_c=cols/2
    for u in range(0,rows):
        for v in range(0,cols):
            d=sqrt(pow(u-center_r,2)+pow(v-center_c,2))
            new_image[u][v]=1-exp(-(d*d)/(2*d0*d0))

    return shift(new_image)*image

if __name__ == '__main__':
    image=cv2.imread('lena.jpg',0)
    image=cv2.resize(image, (226,226))
    new_image=fourier(image)
    ideal_l=ideal_low_pass_filter(new_image,70)
    ideal_h=ideal_high_pass_filter(new_image,70)
    butterworth_l=butterworth_low_pass_filter(new_image,70,2)
    butterworth_h=butterworth_high_pass_filter(new_image,70,2)
    gauss_l=gauss_low_pass_filter(new_image,70)
    gauss_h=gauss_high_pass_filter(new_image,70)
    
    inverse_fourier(ideal_l,"ideal low")
    inverse_fourier(ideal_h,"ideal high")
    inverse_fourier(butterworth_l,"butterworth low")
    inverse_fourier(butterworth_h,"butterworth high")
    inverse_fourier(gauss_l,"gauss low")    
    inverse_fourier(gauss_h,"gauss high")
    
    cv2.imshow('Imagen original',image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
