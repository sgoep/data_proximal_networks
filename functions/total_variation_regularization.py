import numpy as np

def my_grad(X):
    fx = np.concatenate((X[1:,:],np.expand_dims(X[-1,:], axis=0)), axis=0) - X
    fy = np.concatenate((X[:,1:],np.expand_dims(X[:,-1], axis=1)), axis=1) - X
    return fx, fy

def my_div(Px, Py):
    fx = Px - np.concatenate((np.expand_dims(Px[0,:], axis=0), Px[0:-1,:]), axis=0)
    fx[0,:] = Px[0,:]
    fx[-1,:] = -Px[-2,:]
   
    fy = Py - np.concatenate((np.expand_dims(Py[:,0], axis=1), Py[:,0:-1]), axis=1)
    fy[:,0] = Py[:,0]
    fy[:,-1] = -Py[:,-2]

    return fx + fy


def tv(x0, A, AT, g, alpha, L, Niter, f):
    
    tau = 1/L
    sigma = 1/L
    theta = 1
 
    f = f
    grad_scale = 1 #1e+2
    m, n    = x0.shape
    p    = np.zeros_like(g)
    qx   = x0
    qy   = x0
    u    = x0
    ubar = x0

    error = np.zeros(Niter)
    for k in range(Niter):
        p  = (p + sigma*(A(ubar) - g))/(1+sigma)
        [ubarx, ubary] = my_grad(np.reshape(ubar, [m,n]))
        qx = alpha*(qx + grad_scale*sigma*ubarx)/np.maximum(alpha, np.abs(qx + grad_scale*sigma*ubarx)) 
        qy = alpha*(qy + grad_scale*sigma*ubary)/np.maximum(alpha, np.abs(qy + grad_scale*sigma*ubary))
        
        uiter = np.maximum(0, u - tau*(AT(p) - grad_scale*my_div(qx, qy)))
    
        ubar = uiter + theta*(uiter - u)
        u = ubar
        
        error[k] = np.sum(abs(ubar - f)**2)/np.sum(abs(f)**2)
        # print('TV Iteration: ' + str(k+1) + '/' + str(Niter) + ', Error:' + str(error[k]))
      
    rec = np.reshape(u, [m, n])
    return rec