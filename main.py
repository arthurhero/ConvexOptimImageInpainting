from image_process import *
from matrix_completion import *
from fourier import *

import numpy as np

if __name__ == '__main__':
    # experiment code 

    '''
    percents = [0.1,0.2,0.5,0.9] # percentage of missing pixels
    num_holes = [2,5,10,100,1000,10000] # number of square holes
    num_img = 10
    '''
    percents = [0.1,0.9] # percentage of missing pixels
    num_holes = [2,10000] # number of square holes
    num_img = 10

    for p in range(len(percents)):
        for nh in range(len(num_holes)):
            percent = percents[p]
            num_hole = num_holes[nh]
            print("pencent:",percent)
            print("num_hole:",num_hole)
            imgs, masks = load_dataset('data',num_img,percent,num_hole)
            mc_l1 = 0.0
            fo_l1 = 0.0 # l1 difference between restored and orig img
            for i in range(len(imgs)):
                print("img:",i)
                img = imgs[i]
                #display_img(img)
                mask = masks[i]
                save_img('demo/'+str(i)+'_'+str(p)+'_'+str(nh)+"masked.jpg",(apply_mask(img,mask)*255).astype(int))
                mc_restored_img = matrix_completion(img,mask) # get restored img using matrix completion
                print("mc:",np.sum(np.abs(mc_restored_img-img))/(np.sum(mask)))
                save_img('demo/'+str(i)+'_'+str(p)+'_'+str(nh)+"mc.jpg",(mc_restored_img*255).astype(int))
                fo_restored_img = fourier_compressive_sensing(img,mask) # get restored img using compressive sensing
                print("fo:",np.sum(np.abs(fo_restored_img-img))/(np.sum(mask)))
                save_img('demo/'+str(i)+'_'+str(p)+'_'+str(nh)+"fo.jpg",(fo_restored_img*255).astype(int))
                mc_l1 += np.sum(np.abs(mc_restored_img-img))/(np.sum(mask)) # take average l1 diff
                fo_l1 += np.sum(np.abs(fo_restored_img-img))/(np.sum(mask)) # take average l1 diff
            mc_l1 /= len(imgs)
            fo_l1 /= len(imgs)
            print("mc_l1:",mc_l1)
            print("fo_l1:",fo_l1)

