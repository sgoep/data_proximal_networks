import numpy as np
import matplotlib.pyplot as plt
import random
import os, os.path
import torch
import torch.nn as nn
import torch.optim as optim
# from unet import UNet
from config import config
from numpy import matlib
import matplotlib.pyplot as plt

def window(r, omega, j, wtype):
    
    def ny(x):
        
        def s(x):
            return np.exp(-((1+x)**(-2)+(1-x)**(-2)))
        
        y = np.zeros_like(x)
        case2=(x>0)&(x<1)
        y[case2]= s(x[case2]-1)/(s(x[case2]-1)+s(x[case2]))
        case3=(x>=1)
        y[case3]=1
        return y
        # return 3*x**2 - 2*x**3
    
    def ws_meyer(r):
        u = np.zeros_like(r)
        R = np.abs(r)
        u[R<4/3] = 1;
        case4=(R>=4/3) & (R<5/3)
        u[case4] = np.cos(np.pi/2*ny(3*R[case4]-4))
        return u
        
    def wr_meyer(r):
        u = np.zeros_like(r)
        R = np.abs(r)
        case2=(R>=2/3) & (R<=5/6)
        u[case2] = np.cos(np.pi/2*ny(5-6*R[case2]))
        case3=(R>=5/6) & (R<=4/3)
        u[case3] = 1
        case4=(R>=4/3) & (R<=5/3)
        u[case4] = np.cos(np.pi/2*ny(3*R[case4]-4))
        return u
            
    def wa_meyer(OM):
        v = np.zeros_like(OM)
        case2=(np.abs(OM)>=1/3) & (np.abs(OM)<=2/3)
        v[case2] = np.cos(np.pi/2*ny(3*np.abs(OM[case2])-1));
        case3=(np.abs(OM)<=1/3)
        v[case3] = 1;
        return v
    
    if wtype == 'rs':
        w = ws_meyer(r*2**(-j))
    elif wtype == 'radial':
        w = wr_meyer(r*2**(-j))
    elif wtype == 'angular':
        w = wa_meyer(omega)
    
    return w


def rect2win(crect, N):
    k1, k2 = crect.shape
    c = np.zeros((N, N), dtype='complex')
    k = np.linspace(-N/2, N/2-1, N)
    for i1 in range(k1):
        for i2 in range(k2):
            # c[np.mod(k, k1)==i1, np.mod(k, k2)==i2] = crect[i1, i2]
            mask1 = np.mod(k, k1) == i1
            mask2 = np.mod(k, k2) == i2
            c[mask1.reshape(-1, 1) & mask2] = crect[i1, i2]

    return c

def win2rect(c, k, whichdim):
    N1, N2 = c.shape
    n1, n2 = int(np.log2(N1)), int(np.log2(N2))
    K = int(np.log2(k))
    
    if whichdim == 1:
        for i1 in range(n1-1, K-1, -1):
            c = c[:2**i1, :] + c[2**i1:, :]
    elif whichdim == 2:
        for i2 in range(n2-1, K-1, -1):
            c = c[:, :2**i2] + c[:, 2**i2:]
    return c

def rpc(k1, k2):
    # k1 = np.asarray(k1, dtype = 'int')
    # k2 = np.asarray(k2, dtype = 'int')
    r = np.maximum(abs(k1), abs(k2))
    omega1 = np.arctan2(k1, k2)
    omega = np.zeros(omega1.shape)
    case1 = np.abs(omega1) <= np.pi/4
    omega[case1] = np.tan(omega1[case1])
    case2 = (omega1 >= np.pi/4) & (omega1 < 3*np.pi/4)
    omega[case2] = 2 - 1/np.tan(omega1[case2])#np.sin(omega1[case2])/np.cos(omega1[case2])
    case3 = np.abs(omega1) >= 3*np.pi/4
    omega[case3] = 4 + np.tan(omega1[case3])
    case4 = (omega1 >= -3*np.pi/4) & (omega1 < -np.pi/4)
    omega[case4] = -2 - 1/np.tan(omega1[case4])# np.sin(omega1[case4])/np.cos(omega1[case4])

    return r, omega

def get_curvelets(N, nscales):
    J = int(np.log2(N))
    jmax = J - 1
    jmin = J - nscales
    
    k = np.linspace(-N/2, N/2-1, N)
    k1, k2 = np.meshgrid(k, k)
    r, omega = rpc(k1, k2)
    
    w = window(r, omega, jmin-1, 'rs')
    
    Curvelets = {}
    Normalization = {}
    Scaling = {}
    
    Curvelets[(0,0)]      = w
    Normalization[(0, 0)] = N/np.sqrt(np.sum(w**2))
    for j in range(jmin, jmax+1):
        Ntheta = int(2**(np.ceil((j-jmin)/2)))
        wr = window(r, None, j, 'radial')
        for ell in range(-3*Ntheta, 5*Ntheta):
            w = wr*window(None, np.mod(omega*Ntheta-ell+3, 8*Ntheta)-3, j, 'angular')
            Curvelets[(j, ell)]     = w
            Normalization[(j, ell)] = N/np.sqrt(np.sum(w**2))
            
    return Curvelets, Normalization


def ct(f, nscales):
    N = f.shape[0]
    J = int(np.log2(N))
    jmax = J - 1
    jmin = J - nscales
    
    k = np.linspace(-N/2, N/2-1, N)
    k1, k2 = np.meshgrid(k, k)
    r, omega = rpc(k1, k2)
    
    fhat = np.fft.fftshift(np.fft.fft2(f))
    w = window(r, omega, jmin-1, 'rs')
    # w = w * (N/np.sqrt(np.sum(w**2)))
    c = w * fhat
    Ka = min(N, 2**(jmin+1))
    cnew = np.fft.ifftshift(c[N//2-Ka//2:N//2+Ka//2,N//2-Ka//2:N//2+Ka//2])
    cnew = np.fft.ifftshift(c)
    ca = (Ka*Ka/N**2)
    ca = 1
    
    H = {}
    H[(0,0)] = w
    C = {}
    C[(0,0)] = np.fft.ifft2(cnew * ca)
    for j in range(jmin, jmax+1):
        Ntheta = int(2**(np.ceil((j-jmin)/2)))
        wr = window(r, None, j, 'radial')
        for ell in range(-3*Ntheta, 5*Ntheta):
            w = wr*window(None, np.mod(omega*Ntheta-ell+3, 8*Ntheta)-3, j, 'angular')
            # w = w*(N/np.sqrt(np.sum(w**2)))
            c = w*fhat
            if ell == -3*Ntheta:
                Ka = min(N // 2, 2**(j + 1))
                Kb = min(N // 2, 2**(j + 1))
                cnew = c[N//2 - Ka :N//2, N//2 - Kb:N//2]
            elif ell in np.arange(-3*Ntheta+1, -Ntheta):
                Ka = min(N//2, 2**(j+1))
                Kb = min(N//2, 2**(j+2)//Ntheta)
                caux = c[:,N//2-Kb:N//2]
                cnew = win2rect(caux.T, Ka, 2).T
            elif ell == -Ntheta:
                Ka = min(N//2, 2**(j+1))
                Kb = min(N//2, 2**(j+1))
                cnew = c[N//2:N//2+Ka, N//2-Kb:N//2]
            elif ell in np.arange(-Ntheta+1, Ntheta):
                Ka = min(N//2, 2**(j+1))
                Kb = min(N//2, 2**(j+2)//Ntheta)
                caux = c[N//2:N//2+Ka,:]
                cnew = win2rect(caux.T, Kb, 1).T
            elif ell == Ntheta:
                Ka = min(N // 2, 2**(j + 1))
                Kb = min(N // 2, 2**(j + 1))
                cnew = c[N//2:N//2+Ka, N//2:N//2+Kb]
            elif ell in np.arange(Ntheta+1, 3*Ntheta):
                Ka = min(N//2, 2**(j+1))
                Kb = min(N//2, 2**(j+2)//Ntheta)
                caux = c[:,N//2:N//2+Kb]
                cnew = win2rect(caux.T, Ka, 2).T
            elif ell == 3*Ntheta:
                Ka = min(N//2, 2**(j+1))
                Kb = min(N//2, 2**(j+1))
                cnew=c[N//2-Ka:N//2, N//2:N//2+Kb]
            elif ell in np.arange(3*Ntheta+1, 5*Ntheta):
                Ka = min(N//2, 2**(j+1))
                Kb = min(N//2, 2**(j+2)//Ntheta)
                caux = c[N//2-Ka:N//2,:]
                cnew = win2rect(caux.T, Kb, 1).T
            
            H[(j, ell)] = w
            ca = Ka*Kb/N**2
            ca = 1
            C[(j, ell)] = np.fft.ifft2(cnew*ca)
            
    return C, H

def ict(C, nscales, N):
    J = int(np.log2(N))
    jmax = J - 1
    jmin = J - nscales
    
    k = np.linspace(-N/2, N/2-1, N)
    k1, k2 = np.meshgrid(k, k)
    r, omega = rpc(k1, k2)
    
    w = window(r, omega, jmin-1, 'rs')
    # w = w*(np.sqrt(np.sum(w**2))/N)
    c = rect2win(np.fft.fft2(C[(0,0)]), N)
    # ca = N**2/np.prod(C[(0,0)].shape)
    ca = 1
    
    c = np.fft.fftshift(np.fft.fft2(C[(0,0)]))
    
    fhat = w * c * ca
    for j in range(jmin, jmax+1):
        Ntheta = int(2**(np.ceil((j-jmin)/2)))
        wr = window(r, None, j, 'radial')
        for ell in range(-3*Ntheta, 5*Ntheta):
            w = wr*window(None, np.mod(omega*Ntheta-ell+3, 8*Ntheta)-3, j, 'angular')
            # w = w*(np.sqrt(np.sum(w**2))/N)
            calt = np.fft.fft2(C[(j,ell)])
            Ka, Kb = calt.shape
            c = np.zeros([N, N], dtype='complex')
            if ell == -3*Ntheta:
                c[N//2-Ka:N//2, N//2-Kb:N//2] = calt
                ca = N**2/(Ka*Kb)
            elif ell in range(-3*Ntheta+1, -Ntheta):
                caux = matlib.repmat(calt.T, 1, N//Ka).T
                c[:,N//2-Kb:N//2] = caux
            elif ell == -Ntheta:
                c[N//2:N//2+Ka, N//2-Kb:N//2] = calt
            elif ell in np.arange(-Ntheta+1, Ntheta):
                caux = matlib.repmat(calt.T, N//Kb, 1).T
                c[N//2:N//2+Ka,:] = caux
            elif ell == Ntheta:
                c[N//2:N//2+Ka, N//2:N//2+Kb] = calt
            elif ell in np.arange(Ntheta+1, 3*Ntheta):
                caux = matlib.repmat(calt.T, 1, N//Ka).T
                c[:,N//2:N//2+Kb] = caux
            elif ell == 3*Ntheta:
                c[N//2-Ka:N//2, N//2:N//2+Kb] = calt
            elif ell in np.arange(3*Ntheta+1, 5*Ntheta):
                caux = matlib.repmat(calt.T, N//Kb, 1).T
                c[N//2-Ka:N//2,:] = caux
            # ca = N**2/(Ka*Kb)
            ca = 1
            fhat += w * c * ca
    
    f = np.fft.ifft2(np.fft.ifftshift(fhat))
    f = np.real(f)
    return f

# sigma = 0.05
# f = np.load('data_dfd/frec/frec_' + str(1) + '.npy')
# f[39:49, 95:120] = 1
# np.random.seed(8)
# noise = sigma*np.abs(f).max()*np.random.randn(*f.shape)
# fnoise = f + noise

# from my_radon import filter_projection
# g = np.zeros([120, 200])
# gnoise = sigma*np.random.randn(*g.shape)
# gnoise = filter_projection(gnoise)
# # noise = np.reshape(R.T @ gnoise.flatten(), [128, 128])


# C, H = ct(noise, 4)
# Cflat = []
# Cnoise = []
# for key in C:
#     if key[0] != 0:
#         Cflat.append(C[key].flatten())
#     if key[0] == 3:
#         Cnoise.append([0.4]*len(C[key].flatten()))
#     elif key[0] == 4:
#         Cnoise.append([0.4/2**(4/2)]*len(C[key].flatten()))
#     elif key[0] == 5:
#         Cnoise.append([0.4/2**(5/2)]*len(C[key].flatten()))
#     elif key[0] == 6:
#         Cnoise.append([0.4/2**(6/2)]*len(C[key].flatten()))
# # Cflat = [C[key].numpy().flatten() for key in C]
# Cflat  = np.concatenate( Cflat,  axis=0 )
# Cnoise = np.concatenate( Cnoise, axis=0 )
# # Cflat.sort()
# plt.figure()
# plt.plot(Cflat)
# # plt.plot([np.max((Cflat))]*len(Cflat))
# plt.plot(Cnoise)
# plt.plot(-Cnoise)