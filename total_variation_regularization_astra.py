# %%
import numpy as np
import astra
import torch
import os 
from misc.radon_operator import ram_lak_filter, get_matrix
from config import config

def my_grad(X):
    fx = torch.cat((X[1:,:], X[-1,:].unsqueeze(0)), dim=0) - X
    fy = torch.cat((X[:,1:], X[:,-1].unsqueeze(1)), dim=1) - X
    return fx, fy

def my_div(Px, Py):
    fx = Px - torch.cat((Px[0,:].unsqueeze(0), Px[0:-1,:]), dim=0)
    fx[0,:] = Px[0,:]
    fx[-1,:] = -Px[-2,:]
   
    fy = Py - torch.cat((Py[:,0].unsqueeze(1), Py[:,0:-1]), dim=1)
    fy[:,0] = Py[:,0]
    fy[:,-1] = -Py[:,-2]

    return fx + fy

def tv(x0, A, g, alpha, L, Niter, f):

    tau = 1/L
    sigma = 1/L
    theta = 1
    grad_scale = 1e+2
    # m, n    = x0.shape
    m, n = f.shape
    Nal, Ns = g.shape
    try:
        p    = torch.Tensor(np.zeros_like(g)).to(config.device).to(torch.float)
    except:
        p    = torch.Tensor(g).to(config.device).to(torch.float)
    qx   = x0
    qy   = x0
    u    = x0
    ubar = x0

    alpha = torch.Tensor([alpha]).to(config.device)
    zero_t = torch.Tensor([0]).to(config.device)

    error = torch.zeros(Niter)
    for k in range(Niter):

        p  = (p + sigma*(torch.matmul(A, ubar.reshape(-1, 1)).reshape(Nal, Ns) - g))/(1+sigma)
        
        # p = p.to(torch.float)
        ubarx, ubary = my_grad(torch.reshape(ubar, [m, n]))
        if alpha > 0:
            qx = alpha*(qx + grad_scale*sigma*ubarx)/torch.maximum(alpha, torch.abs(qx + grad_scale*sigma*ubarx)) 
            qy = alpha*(qy + grad_scale*sigma*ubary)/torch.maximum(alpha, torch.abs(qy + grad_scale*sigma*ubary))
            # print(type(AT(p)))
            uiter = torch.maximum(zero_t, u - tau*(torch.matmul(A.T, p.reshape(-1, 1)).reshape(m, n) - grad_scale*my_div(qx, qy)))
        else:
            uiter = torch.maximum(zero_t, u - tau*torch.matmul(A.T, p.reshape(-1, 1)).reshape(m, n))
    
        ubar = uiter + theta*(uiter - u)
        u = ubar
        
        error[k] = torch.sum(torch.abs(ubar - f)**2)/torch.sum(torch.abs(f)**2)
        print('TV Iteration: ' + str(k+1) + '/' + str(Niter) + ', Error: ' + str(error[k]))
      
    rec = torch.reshape(u, [m, n])
    return rec


if __name__ == "__main__":
    index = 1

    X = np.load(f"./data/phantom/phantom_{str(index)}.npy")
    N = 128
    Nal = 180
    Ns = 200
    al_full = np.linspace(-np.pi/2, np.pi/2*(1-1/Nal), Nal, endpoint=True)
    Phi = np.pi/3
    al1 = al_full[abs(al_full)<=Phi]
    A1 = get_matrix(N, Ns, al1)

    X = torch.Tensor(X)
    g1 = A1.matmul(X.reshape(-1, 1)).reshape(len(al1), Ns)

    delta = 0.03
    eta   = torch.abs(g1).max()*torch.randn(*g1.shape)
    noise = delta*eta
    gnoise = g1 + noise


    import matplotlib.pyplot as plt
    fbp = A1.T.matmul(ram_lak_filter(gnoise).reshape(-1, 1)).reshape(N, N)
    plt.figure()
    plt.imshow(fbp)
    plt.colorbar()

    x0 = torch.zeros_like(X)
    alpha = 0.1
    L = 200
    Niter = 1000
    rec = tv(x0, A1, gnoise, alpha, L, Niter, X)

    plt.figure()
    plt.imshow(rec)
    plt.colorbar()

# X = torch.Tensor(X).to("cuda")
# # xx, xy = my_grad(X)

# # rec = my_div(xx, xy)

# # plt.figure()
# # plt.imshow(rec)
# # plt.colorbar()

# def get_radon_operator(N1, N2, Ns, al, pixel_width=1):
#     volumeGeometry = astra.create_vol_geom(N1, N2)
#     projectionGeometry = astra.create_proj_geom('parallel', pixel_width, Ns, al)
#     proj_id = astra.create_projector('line', projectionGeometry, volumeGeometry)
#     A = astra.OpTomo(proj_id)
#     return A, proj_id, volumeGeometry, projectionGeometry


# def get_radon_matrix(N1, N2, Ns, al, pixel_width=1):
#     _, proj_id, _, _ = get_radon_operator(N1, N2, Ns, al, pixel_width=1)
#     mat_id = astra.projector.matrix(proj_id)
#     Amat = astra.matrix.get(mat_id)
#     return Amat

# def filter_sinogram(g):
#     a, b = g.shape
#     # filter = np.append( np.linspace(max_al, 0, b//2, endpoint=False), np.linspace(0, max_al, b//2, endpoint=False) )
#     filter = np.append( np.linspace(np.pi/2, 0, b//2, endpoint=False), np.linspace(0, np.pi/2, b//2, endpoint=False) )
#     filter = filter[None,:]
#     gfilt = np.zeros_like(g, dtype="complex")
#     for i in range(a):
#         ghat = np.fft.fftshift(np.fft.fft(g[i,:]))/(g.shape[0])
#         gfilt[i,:] = np.fft.ifft(np.fft.ifftshift(ghat* filter))# * filter)
#     return np.real(gfilt)

# Nal = 180
# Phi = np.pi/4
# al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
# al1 = al_full[np.abs(al_full)<=Phi]

# Ns = 200
# Nal1 = len(al1)

# _, _, volGeom, projGeom = get_radon_operator(X.shape[0], X.shape[1], Ns, al1)

# ds = projGeom["DetectorWidth"]/projGeom["DetectorCount"]
# dtheta = al1[1] - al1[0]

# dx = 2*volGeom["option"]["WindowMaxX"]/volGeom["GridRowCount"]

# A = get_radon_matrix(X.shape[0], X.shape[1], Ns, al1).toarray()

# A = torch.Tensor(A).to("cuda")

# # norms = np.linalg.norm(A, axis=0)
# # A = A / norms

# delta = 0.03

# forward = lambda x: torch.matmul(A, x.reshape(-1, 1)).reshape(len(al1), Ns)
# adjoint = lambda y: torch.matmul(A.T, y.reshape(-1, 1)).reshape(image_size, image_size)#*ds*dtheta

# g = forward(X)

# eta   = np.abs(g.cpu().numpy()).max()*np.random.randn(*g.cpu().numpy().shape)
# noise = delta*eta
# gnoise = g + torch.Tensor(noise).to("cuda")

# # gf = torch.Tensor(filter_sinogram(gnoise))

# # fbp = adjoint(gf)


# alpha = 0.05
# L = 200

# # if not L:
#     # L = scipy.sparse.linalg.eigsh(2*np.dot(D.T, D), 1, which='LM')[0]

# # if os.path.exists("largest_singular_value.pt"):
# #     L = torch.load("largest_singular_value.pt")
# # else:
# #     L = torch.linalg.svdvals(2*torch.matmul(A.T, A))[0]
# #     torch.save(L, "largest_singular_value.pt")

# Niter = 1000

# x0 = torch.zeros_like(X).to("cuda")

# rec = tv(x0, forward, adjoint, gnoise, alpha, L, Niter, X)

# rec = rec.cpu().numpy()
# print(rec)

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(rec)
# # plt.show()
# plt.colorbar()
# plt.savefig('foo.pdf')


# # %%


# %%
