import numpy as np
from co_utils import *
import scipy.fftpack as spfft

def fourier_compressive_sensing(img,mask):
    '''
    use compressive sensing in fourier domain to restore bad pixels in img
    img - h x w x c, img to be restored
    mask - h x w x 1, 0 for good pixel, 1 for bad pixel
    return restored img
    '''
    h,w,c = img.shape
    
    '''
    suppose X is img in frequency domain, A is the inverse discrete
    cosine transformation, Y is the bad img, Y' is the restored img.
    we have Y'=AX, Y'=Y at good pixels, while minimizing l1(X)
    '''
    mask_flat = mask[:,:,0].T.flatten() # stack the cols into a vector
    idx = (1.-mask_flat).nonzero() # idx for good pixels
    
    # create inverse dct matrix operator 
    idctw = spfft.idct(np.identity(w), norm='ortho', axis=0) # W
    idcth = spfft.idct(np.identity(h), norm='ortho', axis=0) # H

    '''
    We have Y'=H((WX^T)^T)=HXW^T. This is equal to A(flat(X)) where A=kron(W,H). But A is too large, thus
    we directly use H and W.
    For (ij)-th element of matrix (HXW^T)_ij, it is simply (H_ri)X(W^T)_cj, ri means i-th row, cj means j-th col
    '''

    # variable for calculating the gradient. See explanation in next comment block.
    print("get hw")
    HW = np.outer(idcth.T,idctw)
    print("got outer")
    print(idcth.flatten().reshape(-1,1).shape)
    #HW = np.matmul(idcth.flatten().reshape(-1,1),idctw.flatten().reshape(1,-1)).reshape(h,h,w,w)
    print("got hw")
    #HW = np.transpose(HW, (1,2,0,3)).reshape(h,w,h*w)

    l1_lambda = 10.
    lr = 10.

    # we process channel by channel
    img_restored = np.zeros_like(img)
    for i in range(c):
        img_c = img[:,:,i]
        img_flat = img_c.T.flatten()
        img_sampled = img_flat[idx]

        '''
        since the feasible set constraint is on y not on X, it is hard to compute the feasible set for X, thus
        we instead try to minimize |S(Y')-S(Y)|^2+lambda|X|_1 using proximal grad
        S is the sampling operator, suppressing invalid position to 0

        we want to calculate the grad of ((H_ri)X(W^T)_cj-Y_ij)^2
        which is just 2*((H_ri)X(W^T)_cj-Y_ij)*outer(H_ri,W_rj)
        and the total gradient is the sum for all valid pixels
        Thus, we first calculate (flat(H))^T*(flat(W)), then fold the shape from (h*h)x(w*w) to h x w x (h*w)
        Then we reshape S(HXW^T-Y) to be 1 x 1 x (h*w)
        Finally we time the two tensors together, and sum along the last axis
        '''

        def grad(x):
            diff = np.matmul(H,np.matmul(x,W.T))
            diff_sampled = diff*(1-mask[:,:,0])
            diff_flat = diff_sampled.reshape(1,1,h*w)
            gradient = np.sum(diff_flat*HW,axis=-1) # h x w

        prox = lambda x : (x>0).astype(float)*np.maximum(0.,np.abs(x)-lr*l1_lambda) # proximal gradient of l1
        
        x_best = fista(np.zeros((h,w)),grad,prox,lr=lr) # best representation in the spectral domain
        img_restored[:,:,i]  = np.matmul(H,np.matmul(x_best,W.T)) # h x w
    return img_restored*mask+img*(1-mask)


