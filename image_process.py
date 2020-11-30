import cv2
import numpy as np
import random
import os

def load_img(fname):
    '''
    return img in h x w x c, float
    '''
    img = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
    img = img.astype(float)
    img /= 255.0
    return img

def save_img(fname,img):
    cv2.imwrite(fname,img)

def display_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bbox_img(img,y1,x1,y2,x2,color=(1.,1.,1.)):
    '''
    draw a box filled with color on img
    '''
    ret=img.copy()
    cv2.rectangle(ret,(x1,y1),(x2,y2),color,-1)
    return ret

def produce_mask(h,w,percent,num):
    '''
    h,w - h and w of the mask
    percent - percentage of missing pixels
    num - how many square holes
    return mask h x w x 1
    '''
    side = int(np.power(float(h*w)*percent/num,0.5)) # approximate side of the square holes
    if side==0:
        side = 1
    mask = np.zeros((h,w,1))
    for i in range(num):
        # randomly generate top left point for each square hole
        x1=random.randint(0,w-side)
        y1=random.randint(0,h-side)
        mask = bbox_img(mask, y1,x1,y1+side,x1+side,(1.0,))
    return mask

def apply_mask(img,mask):
    '''
    img - h x w x c
    mask - h x w x 1
    return masked img (filled with white)
    '''
    return img*(1-mask)+mask

def load_dataset(path,num,percent,num_hole):
    '''
    path - path to the img folder
    num - number of imgs to load (imgs must be named 1.jpg, 2.jpg, etc.)
    percent - percentage of missing pixels
    num_hole - how many square holes
    '''
    random.seed(0)
    imgs = list()
    masks = list()
    for i in range(1,num+1):
        img = load_img(os.path.join(path,str(i)+'.jpg'))
        h,w,_ = img.shape
        mask = produce_mask(h,w,percent,num_hole)
        imgs.append(img)
        masks.append(mask)
        #display_img(apply_mask(img,mask))
    return imgs, masks

if __name__ == '__main__':
    '''
    mask = produce_mask(256,256,0.25,3000)
    display_img(mask)
    '''
    load_dataset('data',5,0.9,1000)
