import numpy as np
from co_utils import *
import scipy.fftpack as spfft
from image_process import display_img

def fourier_compressive_sensing(img,mask):
    '''
    use compressive sensing in fourier domain to restore bad pixels in img
    img - h x w x c, img to be restored
    mask - h x w x 1, 0 for good pixel, 1 for bad pixel
    return restored img
    '''
    print("begin  fourier")
    h,w,c = img.shape
    
    '''
    suppose X is img in frequency domain, A is the inverse discrete
    cosine transformation, Y is the bad img, Y' is the restored img.
    we have Y'=AX, Y'=Y at good pixels, while minimizing l1(X)
    '''
    mask_flat = mask[:,:,0].T.flatten() # stack the cols into a vector
    mask_c = mask[:,:,0] # stack the cols into a vector
    idx = (1.-mask_flat).nonzero() # idx for good pixels
    
    # create dct matrix operator 
    dctw = spfft.dct(np.identity(w), norm='ortho', axis=0)
    dcth = spfft.dct(np.identity(h), norm='ortho', axis=0)

    # create inverse dct matrix operator 
    W = spfft.idct(np.identity(w), norm='ortho', axis=0) # W
    H = spfft.idct(np.identity(h), norm='ortho', axis=0) # H

    '''
    fr = np.matmul(dcth,np.matmul(img[:,:,0],dctw.T))
    r = np.matmul(H,np.matmul(fr,W.T))
    display_img(img[:,:,0])
    display_img(fr)
    display_img(r)
    '''

    '''
    We have Y'=H((WX^T)^T)=HXW^T. This is equal to A(flat(X)) where A=kron(W,H). But A is too large, thus
    we directly use H and W.
    For (ij)-th element of matrix (HXW^T)_ij, it is simply (H_ri)X(W^T)_cj, ri means i-th row, cj means j-th col
    '''

    # variable for calculating the gradient. See explanation in next comment block.
    '''
    print("get hw")
    #HW = np.outer(idcth.T,idctw)
    print("got outer")
    print(H.flatten().reshape(-1,1).shape)
    HW = np.matmul(H.flatten().reshape(-1,1),W.flatten().reshape(1,-1))
    print("got hw")
    HW = HW.reshape(h,h,w*w).transpose(0,2,1).reshape(h,w,w,h).transpose(0,1,3,2).reshape(h*w,h,w).transpose(1,2,0)
    print(HW.shape)
    print("diff",HW[:,:,0]-np.matmul(H[0].reshape(h,1),W[0].reshape(1,w)))
    print("diff",HW[:,:,15]-np.matmul(H[0].reshape(h,1),W[15].reshape(1,w)))
    print("diff",HW[:,:,h+30]-np.matmul(H[1].reshape(h,1),W[30].reshape(1,w)))
    '''


    l1_lambda = 0.01
    lr = .1

    # we process channel by channel
    img_restored = np.zeros_like(img)
    for i in range(c):
        img_c = img[:,:,i]
        img_sampled = img_c*(1-mask_c) 
        img_s_f = np.matmul(dcth,np.matmul(img_sampled,dctw.T)) # damaged img in fourier domain

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
            '''
            grad of |S(Ax)-S(y)|^2: 2*A^T(AX-y)
            '''
            print("get diff")
            diff = np.matmul(H,np.matmul(x,W.T))-img_sampled # AX-y
            print("diff",diff[0][0])
            diff_sampled = diff*(1-mask_c) # S(AX-y)
            gradient = 2*np.matmul(dcth,np.matmul(diff_sampled,dctw.T)) # 2*A^T(AX-y) since dct is orthogonal
            print("grad",gradient[0][0])
            '''
            diff_flat = diff_sampled.reshape(1,1,h*w)
            print("got diff")
            print(diff_flat.shape)
            prod = diff_flat*HW
            print("got prod",prod.shape)
            gradient = np.sum(prod,axis=-1) # h x w
            print("got sum",gradient.shape)
            '''
            return gradient

        prox = lambda x : 2*((x>0).astype(float)-0.5)*np.maximum(0.,np.abs(x)-lr*l1_lambda) # proximal gradient of l1
        
        #x_best = fista(img_s_f,grad,prox,lr=lr) # best representation in the spectral domain
        x_best = fista(np.zeros_like(img_c),grad,prox,lr=lr) # best representation in the spectral domain
        img_restored[:,:,i]  = np.matmul(H,np.matmul(x_best,W.T)) # h x w
        display_img(img_restored)
    return img_restored*mask+img*(1-mask)


