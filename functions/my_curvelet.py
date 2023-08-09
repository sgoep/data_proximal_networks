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

def right(omega):
    r = np.zeros_like(omega)
    idx1 = (1/3 <= omega) & (omega <= 2/3)
    xx = 3*np.abs(omega[idx1]) - 1
    r[idx1] = np.cos(np.pi/2 * (3*(xx**2) - 2*(xx**3)))
    return r
    
def left(omega):
    l = np.zeros_like(omega)
    idx1 = (-1/3 >= omega) & (omega >= -2/3)
    xx = 3*np.abs(omega[idx1]) - 1
    l[idx1] = np.cos(np.pi/2 * (3*(xx**2) - 2*(xx**3)))
    return l

def middle(omega, Phi):
    m = np.zeros_like(omega)
    idx1 = (np.abs(omega) <= Phi)
    m[idx1] = 1
    return m

def window2(omega, denom, shift, b, Phi):
    w = left((omega+shift)/denom * b)+ middle(omega, Phi)+ right((omega-shift)/denom * b)
    return w

def cart2pol(x, y):
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return(rho, phi)

def get_curvelets_adapted(N, Phi, nscales):
    Phi_vis = np.deg2rad(Phi)
    # Phi_vis = np.pi/4
    Phi_inv = np.pi/2 - Phi_vis
    
    J = int(np.log2(N))
    jmax = J - 1
    jmin = J - nscales
    
    k = np.linspace(-N/2, N/2, N)
    k2, k1 = np.meshgrid(k, k)
    r, _ = rpc(k1, k2)
    _, omega = cart2pol(k1, k2)
    # omega = omega.T
        
    w = window(r, omega, jmin-1, 'rs')
    
    Curvelets = {}
    Normalization = {}
    Scaling = {}
    
    Curvelets[(0,0)]      = w
    Normalization[(0, 0)] = N/np.sqrt(np.sum(w**2))
    for j in range(jmin, jmax+1):
        Ntheta_vis = int(2**(np.ceil((j-jmin)/2)+1))
        Ntheta_inv = int(2**(np.ceil((j-jmin)/2)+1))
        wr = window(r, None, j, 'radial')
        for ell in range(-Ntheta_vis//2, Ntheta_vis//2):
            # w = wr*window(None, np.mod( omega , 4*Ntheta)-3, j, 'angular')
            
            # VISIBLE PART
            w = wr*window(None, omega/(2*Phi_vis)*Ntheta_vis-ell-0.5, j, 'angular')
            Curvelets[(j, ell, 'b')]     = w
            Normalization[(j, ell, 'b')] = N/np.sqrt(np.sum(w**2))

            w = wr*window(None, np.flipud(omega)/(2*Phi_vis)*Ntheta_vis-ell-0.5, j, 'angular')
            Curvelets[(j, ell, 't')]     = w
            Normalization[(j, ell, 't')] = N/np.sqrt(np.sum(w**2))
            
        for ell in range(-Ntheta_vis//2, Ntheta_vis//2):
            
            # # INVISIBLE PART
            if j == jmin:
                h = 0.5
            else:
                h = 0.5
                
            w = wr*window2((omega + np.pi/2)*Ntheta_inv - (h + ell) * 2 * Phi_inv, 2*Phi_inv, Phi_inv - Phi_vis, Phi_inv/Phi_vis, Phi_inv-1/3*Phi_vis)
            Curvelets[(j, ell, 'l')]     = w
            Normalization[(j, ell, 'l')] = N/np.sqrt(np.sum(w**2))
            
            w = wr*window2((omega - np.pi/2)*Ntheta_inv - (h + ell) * 2 * Phi_inv, 2*Phi_inv, Phi_inv - Phi_vis, Phi_inv/Phi_vis, Phi_inv-1/3*Phi_vis)
            Curvelets[(j, ell, 'r')]     = w
            Normalization[(j, ell, 'r')] = N/np.sqrt(np.sum(w**2))

    return Curvelets

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
    
    Curvelets[(0,0)]      = w
    Normalization[(0, 0)] = N/np.sqrt(np.sum(w**2))
    for j in range(jmin, jmax+1):
        Ntheta = int(2**(np.ceil((j-jmin)/2)))
        wr = window(r, None, j, 'radial')
        for ell in range(-3*Ntheta, 5*Ntheta):
            w = wr*window(None, np.mod(omega*Ntheta-ell+3, 8*Ntheta)-3, j, 'angular')
            Curvelets[(j, ell)]     = w
            Normalization[(j, ell)] = N/np.sqrt(np.sum(w**2))
            
    return Curvelets
    
def ct(f, W, subsampled):
    C = {}
    fhat = np.fft.fftshift(np.fft.fft2(f))
    for keys in W:
        C[keys] = np.fft.ifft2(fhat * W[keys])
    return C

def ict(C, W, N):
    fhat = np.zeros([N, N], dtype='complex')
    for keys in W:
        fhat += W[keys] * np.fft.fft2(C[keys])
    frec = np.fft.ifft2(np.fft.fftshift(fhat))
    return np.real(frec)

class my_curvelet():
    def __init__(self, N, adapted, is_subsamp, nscales, to_device):
        super(my_curvelet, self).__init__()
        
        self.PHI        = 60
        self.N          = N
        self.J          = int(np.log2(N))
        self.jmax       = self.J - 1
        self.jmin       = self.J - nscales
        self.is_subsamp = is_subsamp
        self.nscales    = nscales
        self.to_device  = to_device
        if adapted:
            tiling = get_curvelets_adapted(N, self.PHI, nscales)
        else:
            tiling = get_curvelets(N, nscales) 
        if to_device:
            self.tiling = {key: torch.Tensor(tiling[key]).to(config.device) for key in tiling}
        else: 
            self.tiling = {key: torch.Tensor(tiling[key]) for key in tiling}
            
        self.adapted    = adapted
               
        k = np.linspace(-N/2, N/2, N)
        k2, k1 = np.meshgrid(k, k)
        r, _ = rpc(k1, k2)
        _, omega = cart2pol(k1, k2)
        chi = torch.zeros([N, N], dtype=torch.bool)
        chi[abs(omega)<=np.deg2rad(self.PHI)] = True
        chi += np.flipud(chi)
        self.chi = chi
        # chi[ chi != 0] = True
        if to_device:
            self.chi = self.chi.to(config.device)
        self.invis_keys = []
        self.vis_keys = []
        for keys in self.tiling:
            if keys != (0,0) and torch.sum(torch.logical_and(self.tiling[keys].to(dtype=torch.bool), self.chi)) <= 0:
                self.invis_keys.append(keys)
            else:
                self.vis_keys.append(keys)
    
    def rect2win(self, crect, N):
        k1, k2 = crect.shape
        c = torch.zeros((N, N), dtype=torch.cdouble)
        if self.to_device:
            c = c.to(config.device)
        k = np.linspace(-N/2, N/2-1, N)
        for i1 in range(k1):
            for i2 in range(k2):
                # c[np.mod(k, k1)==i1, np.mod(k, k2)==i2] = crect[i1, i2]
                mask1 = np.mod(k, k1) == i1
                mask2 = np.mod(k, k2) == i2
                c[mask1.reshape(-1, 1) & mask2] = crect[i1, i2]

        return c

    def win2rect(self, c, k, whichdim):
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
    
    
    def subsample(self, c, key):
        N = self.N
        j = key[0]
        if j == 0:
            Ka = min(N, 2**(self.jmin+1))
            cnew = torch.fft.ifftshift(c[N//2-Ka//2:N//2+Ka//2,N//2-Ka//2:N//2+Ka//2])
            return cnew
            
        ell = key[1]
        Ntheta = int(2**(np.ceil((j-self.jmin)/2)))
        if ell == -3*Ntheta:
            Ka = min(N // 2, 2**(j + 1))
            Kb = min(N // 2, 2**(j + 1))
            cnew = c[N//2 - Ka :N//2, N//2 - Kb:N//2]
        elif ell in np.arange(-3*Ntheta+1, -Ntheta):
            Ka = min(N//2, 2**(j+1))
            Kb = min(N//2, 2**(j+2)//Ntheta)
            caux = c[:,N//2-Kb:N//2]
            cnew = self.win2rect(caux.T, Ka, 2).T
        elif ell == -Ntheta:
            Ka = min(N//2, 2**(j+1))
            Kb = min(N//2, 2**(j+1))
            cnew = c[N//2:N//2+Ka, N//2-Kb:N//2]
        elif ell in np.arange(-Ntheta+1, Ntheta):
            Ka = min(N//2, 2**(j+1))
            Kb = min(N//2, 2**(j+2)//Ntheta)
            caux = c[N//2:N//2+Ka,:]
            cnew = self.win2rect(caux.T, Kb, 1).T
        elif ell == Ntheta:
            Ka = min(N // 2, 2**(j + 1))
            Kb = min(N // 2, 2**(j + 1))
            cnew = c[N//2:N//2+Ka, N//2:N//2+Kb]
        elif ell in np.arange(Ntheta+1, 3*Ntheta):
            Ka = min(N//2, 2**(j+1))
            Kb = min(N//2, 2**(j+2)//Ntheta)
            caux = c[:,N//2:N//2+Kb]
            cnew = self.win2rect(caux.T, Ka, 2).T
        elif ell == 3*Ntheta:
            Ka = min(N//2, 2**(j+1))
            Kb = min(N//2, 2**(j+1))
            cnew=c[N//2-Ka:N//2, N//2:N//2+Kb]
        elif ell in np.arange(3*Ntheta+1, 5*Ntheta):
            Ka = min(N//2, 2**(j+1))
            Kb = min(N//2, 2**(j+2)//Ntheta)
            caux = c[N//2-Ka:N//2,:]
            cnew = self.win2rect(caux.T, Kb, 1).T
        return cnew
    
    def upsample(self, calt, key):
        N = self.N
        j = key[0]
        if j == 0:
            c = self.rect2win(calt, N)
            return c
        
        ell = key[1]
        Ntheta = int(2**(np.ceil((j-self.jmin)/2)))
        Ka, Kb = calt.shape
        c = torch.zeros([N, N], dtype=torch.complex128)
        if self.to_device:
            c = c.to(config.device)
        if ell == -3*Ntheta:
            c[N//2-Ka:N//2, N//2-Kb:N//2] = calt
            ca = N**2/(Ka*Kb)
        elif ell in range(-3*Ntheta+1, -Ntheta):
            # caux = matlib.repmat(calt.T, 1, N//Ka).T
            caux = torch.tile(calt.T, (1, N//Ka)).T
            c[:,N//2-Kb:N//2] = caux
        elif ell == -Ntheta:
            c[N//2:N//2+Ka, N//2-Kb:N//2] = calt
        elif ell in np.arange(-Ntheta+1, Ntheta):
            # caux = matlib.repmat(calt.T, N//Kb, 1).T
            caux = torch.tile(calt.T, (N//Kb, 1)).T
            c[N//2:N//2+Ka,:] = caux
        elif ell == Ntheta:
            c[N//2:N//2+Ka, N//2:N//2+Kb] = calt
        elif ell in np.arange(Ntheta+1, 3*Ntheta):
            # caux = matlib.repmat(calt.T, 1, N//Ka).T
            caux = torch.tile(calt.T, (1, N//Ka)).T
            c[:,N//2:N//2+Kb] = caux
        elif ell == 3*Ntheta:
            c[N//2-Ka:N//2, N//2:N//2+Kb] = calt
        elif ell in np.arange(3*Ntheta+1, 5*Ntheta):
            # caux = matlib.repmat(calt.T, N//Kb, 1).T
            caux = torch.tile(calt.T, (N//Kb, 1)).T
            c[N//2-Ka:N//2,:] = caux
        return c
        
    def transform(self, x):
       if not torch.is_tensor(x):
           x = torch.Tensor(x)
           if self.to_device:
               x = x.to(config.device)
       C = {}
       xhat = torch.fft.fftshift(torch.fft.fft2(x))
       if self.to_device:
           xhat = xhat.to(config.device)
       for keys in self.tiling:
           c = xhat * self.tiling[keys]
           if self.is_subsamp:
               c = self.subsample(c, keys)
           C[keys] = torch.fft.ifft2(c)
       return C
    
    def adjoint(self, C):
        xhat = torch.zeros([self.N, self.N], dtype=torch.complex128)#.to(config.device)
        if self.to_device:
            xhat = xhat.to(config.device)
        for keys in self.tiling:
            c = torch.fft.fft2(C[keys])
            if self.is_subsamp:
                c = self.upsample(c, keys)
            xhat += self.tiling[keys] * c
        xrec = torch.fft.ifft2(torch.fft.fftshift(xhat))
        xrec = torch.real(xrec)
        return xrec
    
    def plot_tiling(self):
        W = np.zeros([self.N, self.N])
        for keys in self.tiling:
            W += self.tiling[keys].numpy()
        plt.figure()
        plt.imshow(W)
        plt.colorbar()
        plt.show()


