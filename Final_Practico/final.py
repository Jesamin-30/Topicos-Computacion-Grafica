import numpy as np
from math import *
import math 
import cv2
from skimage.measure import label
import skimage.measure as measure
from PIL import Image

cruz=[[0, 255, 0], [255, 255, 255], [0, 255, 0]]
diamante=[[0, 0, 255, 0, 0],[0, 255, 255, 255, 0],[255, 255, 255, 255, 255],[0, 255, 255, 255, 0],[0, 0, 255, 0, 0]]
#disco=[[0,0,0,255,0,0,0],[0,255,255,255,255,255,0],[0,255,255,255,255,255,0],[255,255,255,255,255,255,255],[0,255,255,255,255,255,0],[0,255,255,255,255,255,0],[0,0,0,255,0,0,0]]
disco=[[0,0,0,0,255,255,255,0,0,0,0],
[0,0,0,255,255,255,255,255,0,0,0],
[0,0,255,255,255,255,255,255,255,0,0],
[0,255,255,255,255,255,255,255,255,255,0],
[255,255,255,255,255,255,255,255,255,255,255],
[255,255,255,255,255,255,255,255,255,255,255],
[255,255,255,255,255,255,255,255,255,255,255],
[0,255,255,255,255,255,255,255,255,255,0],
[0,0,255,255,255,255,255,255,255,0,0],
[0,0,0,255,255,255,255,255,0,0,0],
[0,0,0,0,255,255,255,0,0,0,0]]

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
    for i in range(0,rows):
        for j in range(0,cols):
            if image1[i][j] and image2[i][j]:
                new_image[i][j]=255

    return new_image


def condicional(image,mask):
    rows,cols=image.shape
    new_image=np.zeros((rows,cols),np.uint8)
    #for i in range(0,10):
    image_x=dilatacion(image,mask,1,1)
    inter=interseccion(image,image_x)
    #new_image=complemento(image)

    return complemento(inter)

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

def sobel(image):
    rows,cols=image.shape   
    new_image=np.zeros((rows,cols),np.uint8)
    #Sx=image.copy()
    #Sy=image.copy()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            cont_x=(-1)*int(image[i-1][j-1])+int(image[i-1][j+1])-2*int(image[i][j-1])+2*int(image[i][j+1])-int(image[i+1][j-1])+int(image[i+1][j+1])
            #Sx[i][j]=cont_x

            cont_y=(-1)*int(image[i-1][j-1])+int(image[i+1][j-1])-2*int(image[i-1][j])+2*int(image[i+1][j])-int(image[i-1][j+1])+int(image[i+1][j+1])
            #Sy[i][j]=cont_y
            new_image[i][j]=abs(cont_x+cont_y)
    return new_image

def recortar_image(image):
    row,col=image.shape   
    rowF=int(row/2)
    colF=0
    column=[]
    
    for i in range(0,rowF):
        temp=True
        for j in range(0,col):
            if image[i][j]==255:
                column.append(j)
                temp=False
        if temp==False:
            break
    idx=int(len(column)/2)
    arriba=(i,column[idx])

    column=[]
    for i in range(0,col):
        temp=True
        for j in range(0,rowF):
            if image[j][i]==255:
                column.append(j)
                temp=False
        if temp==False:
            break
    idx=int(len(column)/2)
    lado=(column[idx],i)

    esquina_a=(arriba[0],lado[1])                 
    distanciaX= abs(arriba[1]-esquina_a[1])
    distanciaY= abs(lado[0]-esquina_a[0])
    esquina_b=((esquina_a[0]+2*distanciaY)+40,esquina_a[1]+2*distanciaX)

    #print(esquina_a,esquina_b)
    return (esquina_a,esquina_b)

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
            
if __name__ == '__main__':
    image=cv2.imread('DB/01_01.jpg',0) 
    preprocessing=cv2.resize(image, (384,288))     
    #cv2.imshow('Preprocesamiento',preprocessing) 
    new_image=sobel(preprocessing)
    t_new_image=cv2.threshold(new_image,127,255,cv2.THRESH_BINARY)
    #cv2.imshow('Sobel',t_new_image[1]) 

    #dilatar=dilatacion(t_new_image[1],diamante,2,2)
    dilatar=dilatacion(t_new_image[1],cruz,1,1)
    cv2.imshow('Dilatacion',dilatar) 


    esquina_a,esquina_b=recortar_image(dilatar)
    cara=preprocessing[esquina_a[0]:esquina_b[0],esquina_a[1]:esquina_b[1]]
    cv2.imshow('Detectar rostro',cara) 

    #CERRADURA
    dilata_cara=dilatar[esquina_a[0]:esquina_b[0],esquina_a[1]:esquina_b[1]]
    n_image=dilatacion(dilata_cara,diamante,2,2)
    closing=erosion(n_image,diamante,2,2)
    cv2.imshow('Closing',closing)

    ojos=erosion(closing,disco,5,5)
    cv2.imshow('OJOS',ojos)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
       