import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2


threshold_values = {}
h = [1]

def Hist(img):
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   return y


def regenerate_img(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y


   
def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
           cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i
    
    return m/float(w)


def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) **2) * h[i]
    v /= w
    return v
            

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)
        
        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))
        
        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf)**2
        
        
        if not math.isnan(V2w):
            threshold_values[i] = V2w


def get_optimal_threshold():
    min_V2w = min(threshold_values.itervalues())
    optimal_threshold = [k for k, v in threshold_values.iteritems() if v == min_V2w]
    print ('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]

img=cv2.imread('circuit.jpg',0)
#img=cv2.resize(img, (226,226))
height, width = img.shape
width_cutoff = width // 2
height_cutoff = height // 2

s1 = img[:height_cutoff, :width_cutoff]
s2 = img[:height_cutoff, width_cutoff:]
s3 = img[height_cutoff:, :width_cutoff]
s4 = img[height_cutoff:, width_cutoff:]

h = Hist(s1)
threshold(h)
op_thres = get_optimal_threshold()
res1 = regenerate_img(s1, op_thres)

h = Hist(s2)
threshold(h)
op_thres = get_optimal_threshold()
res2 = regenerate_img(s2, op_thres)

h = Hist(s3)
threshold(h)
op_thres = get_optimal_threshold()
res3 = regenerate_img(s3, op_thres)

h = Hist(s4)
threshold(h)
op_thres = get_optimal_threshold()
res4 = regenerate_img(s4, op_thres)


img[:height_cutoff, :width_cutoff]=res1
img[:height_cutoff, width_cutoff:]=res2
img[height_cutoff:, :width_cutoff]=res3
img[height_cutoff:, width_cutoff:]=res4
plt.imshow(img,cmap="gray")
plt.show()
