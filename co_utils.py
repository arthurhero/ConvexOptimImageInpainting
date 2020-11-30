import numpy as np

def fista(x,grad,prox,proj = None, lr = 1.,delta = 0.0001):
    '''
    convex optim algorithm using proximal gradient and Nesterov's trick
    x - variable to be optimized (optim solution)
    grad - gradient function (grad(X) returns gradient)
    prox - proximal function
    proj - projection function
    lr - learning rate
    '''
    x_last=x.copy()
    x_cur=x.copy()
    cur_iter = 1
    a_last, a_cur  = 0.5, 1.0 # Nesterov sequence
    while True:
        t = (a_last-1)/a_cur
        y = (1+t)*x_cur-t*x_last
        x_new = prox(y-lr*grad(y))
        print("x_new",x_new[0])
        if proj is not None:
            x_new = proj(x_new)
        if np.sum(np.abs(x_new-x_cur))<delta:
            return x_new
        else:
            print(cur_iter,":",np.sum(np.abs(x_new-x_cur)))
            cur_iter+=1
            x_last = x_cur
            x_cur = x_new
            a_last = a_cur
            a_cur = (cur_iter+1)/2.0

