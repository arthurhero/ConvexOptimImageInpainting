import numpy as np
from co_utils import *
import scipy.fftpack as spfft
from image_process import display_img
import time

def fourier_compressive_sensing(img,mask):
    '''
    use compressive sensing in fourier domain to restore bad pixels in img
    img - h x w x c, img to be restored
    mask - h x w x 1, 0 for good pixel, 1 for bad pixel
    return restored img
    '''
    start = time.time()
    #print("begin fourier")
    h,w,c = img.shape
    
    '''
    suppose X is img in frequency domain, 
    Y is the bad img, Y' is the restored img.
    we have Y'=idct(X), S(Y')=S(Y), while minimizing l1(X)
    S is the sampling operator, suppressing invalid position to 0
    '''
    mask_c = mask[:,:,0] # stack the cols into a vector
    
    # create dct matrix operator, dct(Y)=H(YW^T) 
    dctw = spfft.dct(np.identity(w), norm='ortho', axis=0)
    dcth = spfft.dct(np.identity(h), norm='ortho', axis=0)

    # create inverse dct matrix operator, idct(X)=H(XW^T)
    W = spfft.idct(np.identity(w), norm='ortho', axis=0) # W
    H = spfft.idct(np.identity(h), norm='ortho', axis=0) # H

    '''
    experiment with the dct operators
    fr = np.matmul(dcth,np.matmul(img[:,:,0],dctw.T)) # dct(Y), to freq domain
    r = np.matmul(H,np.matmul(fr,W.T)) # idct(X), to color domain
    display_img(img[:,:,0])
    display_img(fr)
    display_img(r)
    '''

    '''
    We have Y'=H((WX^T)^T)=HXW^T. This is equal to A(flat(X)) where A=kron(W,H). But A is too large, thus
    though we calculate using A, we program with H and W.
    '''

    l1_lambda = .03
    lr = .1 # learning rate

    # we process channel by channel
    img_restored = np.zeros_like(img)
    for i in range(c):
        img_c = img[:,:,i]
        img_sampled = img_c*(1-mask_c) 

        '''
        since the feasible set constraint is on y not on X, it is hard to compute the feasible set for X, thus
        we instead try to minimize |S(Y')-S(Y)|^2+lambda|X|_1 using proximal grad
        
        Using A(flat(X)) where A=kron(W,H), we have the grad of |S(AX)-S(Y)|^2
        to be A^T(S(AX-Y)). Since dct is orthogonal, we have A^T=A^{-1}
        Thus, grad is just 2*dct(S(idct(X)-y))
        '''

        def grad(x):
            '''
            grad of |S(Ax)-S(y)|^2: 2*A^T(AX-y)
            '''
            diff = np.matmul(H,np.matmul(x,W.T))-img_sampled # AX-y
            diff_sampled = diff*(1-mask_c) # S(AX-y)
            gradient = 2*np.matmul(dcth,np.matmul(diff_sampled,dctw.T)) # 2*A^T(AX-y) since dct is orthogonal
            return gradient

        prox = lambda x : 2*((x>0).astype(float)-0.5)*np.maximum(0.,np.abs(x)-lr*l1_lambda) # proximal gradient of l1
        
        x_best = fista(np.zeros_like(img_c),grad,prox,lr=lr) # best representation in the spectral domain
        img_restored[:,:,i]  = np.matmul(H,np.matmul(x_best,W.T)) # h x w
    end = time.time()
    #print("time",end-start)
    return img_restored*mask+img*(1-mask)


