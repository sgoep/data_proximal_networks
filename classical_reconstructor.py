# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:50:42 2022

@author: carndt
"""

import torch
# import odl
import numpy as np




# def define_ray_trafo(start_angle, stop_angle):
    
#     det_shape = 560

#     M = 1.348414746992646 


#     DistanceSourceDetector=553.74
#     DistanceSourceOrigin=410.66

#     DistanceDetectorOrigin = DistanceSourceDetector - DistanceSourceOrigin

#     angle_partition = odl.uniform_partition_fromgrid(
#                 odl.discr.grid.RectGrid(np.linspace(0, 360, 721)*np.pi/180))[int(2*start_angle):int(2*stop_angle+1)]

#     effPixel = 0.1483223173330444


#     det_partition = odl.uniform_partition(-M*det_shape/2, M*det_shape/2, det_shape)

#     geometry =  odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, det_partition, 
#                                                         src_radius=DistanceSourceOrigin/effPixel , 
#                                                           det_radius=DistanceDetectorOrigin/effPixel,
#                                                           src_to_det_init=(-1, 0))
    
#     space = odl.discr.discr_space.uniform_discr([-256,-256], [256,256], (512, 512), dtype=np.float32)

#     ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')
    
#     return ray_trafo


# def define_fbp(start_angle, stop_angle):
#     ray_trafo = define_ray_trafo(start_angle, stop_angle)
#     fbp_op = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo)
    
#     return fbp_op

op_norms = {
    90: 308,
    80: 292,
    70: 275,
    60: 256,
    50: 234,
    40: 208,
    30: 179}

def operator_norm(A):
    x0 = np.random.randn(5,512,512)
    
    for i in range(10):
        for j in range(5):
            x0[j] = x0[j]/np.linalg.norm(x0[j])
            x1 = A.adjoint(A(x0[j]))
            x0[j] = x1
    
    return np.sqrt(max([np.linalg.norm(x0[j]) for j in range(5)]))

def tv(x):
    epsilon = 1e-6
    y1 = x[...,1:,:] - x[...,:-1,:]
    y2 = x[...,:,1:] - x[...,:,:-1]
    Y = torch.zeros(x.shape[-2:],dtype=x.dtype,device=x.device)
    Y[:-1,:-1] = y1[...,:,:-1]**2
    Y[:-1,:-1] = Y[:-1,:-1] + y2[...,:-1,:]**2
    Y[:-1,:-1] = torch.sqrt(Y[:-1,:-1] + epsilon)
    Y[:-1,-1] = torch.abs(y1[...,:,-1])
    Y[-1,:-1] = torch.abs(y2[...,-1,:])
    return Y.sum()

#def binary_loss(x, a, b):
#    return ((x-a)*(x-b)).abs().sum()

def perona_malik(x):
    T=0.001
    
    y1 = x[...,1:,:] - x[...,:-1,:]
    y2 = x[...,:,1:] - x[...,:,:-1]
    Y = torch.zeros(x.shape[-2:],dtype=x.dtype,device=x.device)
    Y[:-1,:-1] = y1[...,:,:-1]**2
    Y[:-1,:-1] = Y[:-1,:-1] + y2[...,:-1,:]**2
    #Y[:-1,:-1] = torch.sqrt(Y[:-1,:-1] + epsilon**2)-epsilon
    #Y[:-1,-1] = torch.abs(y1[...,:,-1])
    #Y[-1,:-1] = torch.abs(y2[...,-1,:])
    Y[:-1,-1] = y1[...,:,-1]**2
    Y[-1,:-1] = y2[...,-1,:]**2
    
    
    Y = T*(1 - torch.exp(-Y/T**2))
    #Y = 1/2*T**2*(1 - torch.exp(-1/T**2*Y**2))
    #Y = torch.sqrt(Y + epsilon**2)-epsilon
    return Y.sum()

def binary_loss(x, a, b):
    #return ((x-a)*(x-b)).abs().sum()
    f1 = (x-a)**2/(b-a)**1.5
    f2 = (x-b)**2/(b-a)**1.5
    return (f1*f2).sum()

def alt_grad_solver(A, y, start_angle, stop_angle, alph=100, bet=1000, lr=0.000001, steps=80):
    '''
    An iterative solver for Limited Angle Tomography. Alternating gradient steps w.r.t.
    the data discrepancy term and the penalty terms (Perona Malik and binary loss)

    Parameters
    ----------
    y : TYPE
        Sinogram data.
    start_angle : TYPE
        lower limit of the measured angles.
    stop_angle : TYPE
        upper limit of the measured angles.
    alph : TYPE, optional
        regularization parameter for Perona Malik. The default is 100.
    bet : TYPE, optional
        regularization parameter for the binary loss. The default is 1000.
    lr : TYPE, optional
        learning rate or step size. Must be very low because of the 
        operator norm of the Ray Transform. The default is 0.000001.
    steps : TYPE, optional
        Number of iterations. The default is 80.

    Returns
    -------
    xn : TYPE
        The reconstuction of the phantom.
    loss : TYPE
        The (overall) loss for every step of the iteration.
    discr : TYPE
        The data discrepancy for every step of the iteration.
    pm_loss : TYPE
        The Perona Malik loss for every step of the iteration.
    bin_loss : TYPE
        The binary loss for every step of the iteration.

    '''
    # A = define_ray_trafo(start_angle, stop_angle)
    
    # initial value
    
    xx, yy = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
    xn= 0.006*np.exp(-3*(xx**2+yy**2))
    
    a=torch.tensor([0.0004], requires_grad=True)
    b=torch.tensor([0.005], requires_grad=True)
    
    discr = []
    pm_loss = []
    bin_loss = []
    loss = []
    for i in np.arange(steps):
        # discrepancy step
        
        res  = A.forward(xn) - y
        xn = np.array(xn - lr*A.backward(res))
        
        if np.isnan(xn.sum()):
            print(str(i) + ', after discrepancy step')
        
        xt = torch.tensor(xn, dtype=torch.float, requires_grad=True)
        
        # penalty step
        pmx = perona_malik(xt)
        binx = binary_loss(xt,a,b)
        penalt = alph*pmx + bet*binx
        penalt.backward()
        xt = xt - lr*xt.grad
        
        # binary penalty adjustment
        with torch.no_grad():
            a = a - 0.001/((bet+0.1)*512**2)*a.grad
            b = b - 0.001/((bet+0.1)*512**2)*b.grad
        a.requires_grad=True
        b.requires_grad=True
        
        #print(str(a) + ', ' + str(b))
        
        xn = xt.detach().numpy()
        
        
        if np.isnan(penalt.detach().numpy()):
            print(str(i) + ', penalty value')
        if np.isnan(xn.sum()):
            print(str(i) + ', after penalty step')
        
        #loss
        discr.append(1/2*np.linalg.norm(res)**2)
        pm_loss.append(pmx.detach().numpy())
        bin_loss.append(binx.detach().numpy())
        loss.append(discr[-1] + alph*pm_loss[-1] + bet*bin_loss[-1])
        #loss.append(1/2*np.linalg.norm(res)**2 + penalt.detach().numpy())
        
    xn[xn<0]=0
    
    return xn, loss, discr, pm_loss, bin_loss
    
    
