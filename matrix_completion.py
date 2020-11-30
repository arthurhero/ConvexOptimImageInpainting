import numpy as np
from co_utils import *

def prox_nucleus(x):
    u,s,vh = np.linalg.svd(x,full_matrices=False)
    d = np.maximum(0.,s-1.)
    return np.matmul(np.matmul(u,np.diag(d)),vh)

def matrix_completion(img,mask):
    '''
    use matrix completion to restore bad pixels in img
    img - h x w x c, img to be restored
    mask - h x w x 1, 0 for good pixel, 1 for bad pixel
    return restored img
    '''
    print("begin matrix!")
    h,w,c = img.shape

    '''
    This is a standard matrix completion problem. Let X be restored img,
    X' be the bad img, S be the sampling operator. We want to minimize
    |X|_* s.t. S(X)=S(X')
    We use a combination of proximal gradient (for nucleus norm) and projection (for the constraint)
    '''
    mask_c = mask[:,:,0] # h x w
    img_restored = np.zeros_like(img)
    grad = lambda x : 0 # our g(x) is 0
    prox = prox_nucleus

    # process channel by channel
    for i in range(c):
        img_c = img[:,:,i]
        proj = lambda x : x*mask_c + img_c*(1-mask_c)
        img_restored[:,:,i] = fista(img_c*(1-mask_c),grad,prox,proj) # project to feasible set

    return img_restored*mask+img*(1-mask)  # though in this case, directly return img_restored should be fine
        
