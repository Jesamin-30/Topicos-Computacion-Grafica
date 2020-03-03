import numpy as np
from math import *
import cv2
import numpy
import numpy as np
from numpy.core.umath_tests import inner1d

def shadow_sobel(image,d=4,t=8):
    rows,cols=image.shape
    new_image=np.zeros(image.shape)
    for i in range(d,rows-d):
        for j in range(d,cols-d):              
            s1=((image[i-d][j]-image[i][j])>t and (image[i+d][j]-image[i][j])>t)
            s2=((image[i][j-d]-image[i][j])>t and (image[i][j+d]-image[i][j])>t)

            if(s1 and s2):
                new_image[i][j]=1
            else:
                new_image[i][j]=0    
    
    return new_image
    #cv2.imshow('shadow Sobel',new_image)
    
            
def light_sobel(image,d=4,t=5):
    rows,cols=image.shape
    new_image=np.zeros(image.shape)
    for i in range(d,rows-d):
        for j in range(d,cols-d):              
            l1=((image[i][j]-image[i-d][j])>t and (image[i][j]-image[i+d][j])>t)
            l2=((image[i][j]-image[i][j-d])>t and (image[i][j]-image[i][j+d])>t)

            if(l1 and l2):
                new_image[i][j]=1
            else:
                new_image[i][j]=0    
    
    return new_image
    #cv2.imshow('light Sobel',new_image)

def shadow_noise_Reduction(image,t=7):
    cv2.imshow('shadow Sobel',image)
    rows,cols=image.shape[:2]
    new_image=np.zeros(image.shape)
    ax=2
    ay=2
    for i in range(ax,rows-ax):
        for j in range(ay, cols-ay):
            suma=0   
            for x in range(i-ax,i+ax):
                for y in range(j-ay,j+ay):
                    suma=suma+image[x][y]
            
            if(suma>t):
                new_image[i][j]=1
            else:
                new_image[i][j]=0
    return new_image
    cv2.imshow('shadow noise',new_image)

def light_noise_Reduction(image,t=8):
    cv2.imshow('light Sobel',image)
    rows,cols=image.shape[:2]
    new_image=np.zeros(image.shape)
    ax=2
    ay=2
    for i in range(ax,rows-ax):
        for j in range(ay, cols-ay):
            suma=0   
            for x in range(i-ax,i+ax):
                for y in range(j-ay,j+ay):
                    suma=suma+image[x][y]
            
            if(suma>t):
                new_image[i][j]=1
            else:
                new_image[i][j]=0

    cv2.imshow('light noise',new_image)
    return new_image

def similarity_Measures(image1,image2):
    rows,cols=image1.shape[:2]
    #new_image=np.zeros(image.shape)
    suma=0.0
    for i in range(0,rows):
        for j in range(0,cols):              
            #print(image1[i][j]-image2[i][j])
            suma=suma+abs(image1[i][j]-image2[i][j])

    suma=suma*(1/(rows*cols))

    if(suma>0.0 and suma<5.0):
        print("Similares")
    else:
        print("No son similares")
    print(suma)
    
def HausdorffDist(A,B):
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    #return(dH)
    if(dH>0.0 and dH<5.0):
        print("Similares")
    else:
        print("No son similares")

    print(dH)

if __name__ == '__main__':

    ###-----------input 1-----------------###
    image1=cv2.imread('001_1.bmp',0)  
    image1=cv2.resize(image1, (220,220))    
    img_s1=shadow_sobel(image1)
    img_l1=light_sobel(image1)

    snr1=shadow_noise_Reduction(img_s1)
    lnr1=light_noise_Reduction(img_l1)
    

    cv2.imshow('reduccion shadow',snr1) 
    cv2.imshow('reduccion light',lnr1) 

    ###-----------input 2-----------------###

    image2=cv2.imread('001_2.bmp',0)      
    image2=cv2.resize(image2, (220,220))    
    img_s2=shadow_sobel(image2)
    img_l2=light_sobel(image2)

    snr2=shadow_noise_Reduction(img_s2)
    lnr2=light_noise_Reduction(img_l2)

    cv2.imshow('2snr',snr2) 
    cv2.imshow('2lnr',lnr2) 

    #similarity_Measures(snr1,snr2)
    #similarity_Measures(lnr1,lnr2)
    
    HausdorffDist(snr1,snr2)
    HausdorffDist(lnr1,lnr2)


    cv2.waitKey(0)
    cv2.destroyAllWindows()