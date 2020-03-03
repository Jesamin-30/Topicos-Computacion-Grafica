#python2 morfologia.py
import numpy as np
from math import *
import cv2


cruz=[[0, 255, 0], [255, 255, 255], [0, 255, 0]]
horizontal=[[255,255,255]]
vertical=[[255],[255],[255]]
diamante=[[0, 0, 255, 0, 0],[0, 255, 255, 255, 0],[255, 255, 255, 255, 255],[0, 255, 255, 255, 0],[0, 0, 255, 0, 0]]
cuadrado=[[255,255],[255,255]]

#x,y es el origen de mascara
def dilatacion(image,mask,x,y):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols),np.uint8)
    m_rows=len(mask)
    m_cols=len(mask[0])
    print (m_rows,m_cols)
    for i in range(0,rows):
        for j in range(0,cols):
            if i+x<rows and j+y<cols:
                if image[i+x][j+y]==255:
                    for k in range(0,m_rows):
                        for l in range(0,m_cols):
                            if mask[k][l]==255:
                                if(i+k<rows and j+l<cols):          
                                    new_image[i+k][j+l]=255

    return new_image
    #cv2.imshow('Dilatacion',new_image) 


def erosion(image,mask,x,y):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols),np.uint8)
    m_rows=len(mask)
    m_cols=len(mask[0])
    print (m_rows,m_cols)
    for i in range(0,rows):
        for j in range(0,cols):
            if i+x<rows and j+y<cols:
                cont_i=0
                cont_m=0
                for k in range(0,m_rows):
                    for l in range(0,m_cols):
                        if mask[k][l]==255:
                            cont_m=cont_m+1
                            if(i+k<rows and j+l<cols):          
                                if(image[i+k][j+l]==255):
                                    cont_i=cont_i+1
                if cont_i==cont_m:
                    new_image[i+x][j+y]=255
    return new_image
    #cv2.imshow('Erosion',new_image)

def complemento(image):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols),np.uint8)
    for i in range(0,rows):
        for j in range(0,cols):
            if image[i][j]==255:
                new_image[i][j]=0
            else:
                new_image[i][j]=255
    return new_image

def interseccion(image1,image2):
    rows,cols=image1.shape
    new_image=np.zeros((rows,cols),np.uint8)
    #print(image1)

    #print(image2)

    for i in range(0,rows):
        for j in range(0,cols):
            if image1[i][j] or image2[i][j]:
                new_image[i][j]=255
            #else:
                #new_image[i][j]=255

    return new_image


if __name__ == '__main__':
    image=cv2.imread('sample2.jpg',0)
    #n_image=dilatacion(image,diamante,2,2)
    #n_image=dilatacion(image,cuadrado,1,0)
    #n_image=dilatacion(image,cruz,1,1)
    #n_image=dilatacion(image,vertical,1,0)
    #n_image=dilatacion(image,horizontal,0,1)
    #cv2.imshow('Dilatar',n_image)

    #n_image=erosion(image,diamante,2,2)
    #n_image=erosion(image,cuadrado,1,0)
    #n_image=erosion(image,cruz,1,1)
    #n_image=erosion(image,vertical,1,0)
    #n_image=erosion(image,horizontal,0,1)
    #cv2.imshow('Erosion',n_image)


    #opening
    '''n_image=erosion(image,diamante,2,2)
    opening=dilatacion(n_image,diamante,2,2)
    cv2.imshow('Opening',opening)'''

    #closing
    '''n_image=dilatacion(image,diamante,2,2)
    closing=erosion(n_image,diamante,2,2)
    cv2.imshow('Closing',closing)'''

    #hit or miss
    image1=erosion(image,cruz,1,1)
    complement=complemento(image)
    image2=erosion(complement,horizontal,0,1)
    hit_miss=interseccion(image1,image2)
    cv2.imshow('Hit or miss',hit_miss)

    cv2.imshow('Imagen original',image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()