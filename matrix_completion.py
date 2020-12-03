import numpy as np
from co_utils import *
import time

def prox_nucleus(x,lamb):
    u,s,vh = np.linalg.svd(x,full_matrices=False)
    d = np.maximum(0.,s-lamb)
    return np.matmul(np.matmul(u,np.diag(d)),vh)

def matrix_completion(img,mask):
    '''
    use matrix completion to restore bad pixels in img
    img - h x w x c, img to be restored
    mask - h x w x 1, 0 for good pixel, 1 for bad pixel
    return restored img
    '''
    #print("begin matrix!")
    h,w,c = img.shape

    '''
    This is a standard matrix completion problem. Let X be restored img,
    X' be the bad img, S be the sampling operator. We want to minimize
    |X|_* s.t. S(X)=S(X')
    Since we cannot use proximal gradient for this, and thus 
    we instead minimize |S(X)-S(X')|^2+|X|_*
    For admm version, see below, which is a little bit slower
    '''
    start = time.time()
    lr=1. # learning rate
    nucleus_lamb = 0.1 # nucleus lambda
    mask_c = mask[:,:,0] # h x w
    img_restored = np.zeros_like(img)

    grad = lambda x : (x-img_c)*(1-mask_c) # gradient S(X-X')
    prox = lambda x : prox_nucleus(x,nucleus_lamb)

    # process channel by channel
    for i in range(c):
        img_c = img[:,:,i]
        img_restored[:,:,i] = fista(img_c*(1-mask_c),grad,prox,lr=lr) # use fista to solve

    end = time.time()
    #print("time",end-start)
    return img_restored*mask+img*(1-mask) 
        
def matrix_completion_admm(img,mask):
    '''
    ADMM version.
    use matrix completion to restore bad pixels in img
    img - h x w x c, img to be restored
    mask - h x w x 1, 0 for good pixel, 1 for bad pixel
    return restored img
    '''
    start = time.time()
    h,w,c = img.shape
    mask_c = mask[:,:,0] # h x w
    u = np.zeros_like(mask_c)
    img_restored = np.zeros_like(img)

    '''
    minimize L(x1,x2,u)=||x_1||*+I(x_2)+1/2||x_1-x_2+u||^2
    x_1 = prox_nucleus(x_2-u)
    x_2 = proj(x_1+u)
    u=u+x_1-x_2
    '''
    for i in range(c):
        img_c = img[:,:,i]
        x1=(img_c*(1-mask_c)).copy()
        x2=x1.copy()
        u = np.zeros_like(mask_c)
        while True:
            x1 = prox_nucleus(x2-u,0.1)
            x2 = (x1+u)*mask_c + img_c*(1-mask_c)
            u = u+x1-x2
            #print(np.sum(np.abs(x1-x2)))
            if np.sum(np.abs(x1-x2))<0.001:
                break
        img_restored[:,:,i] = x2
    end= time.time()
    #print("time",end-start)
    return img_restored*mask+img*(1-mask)
