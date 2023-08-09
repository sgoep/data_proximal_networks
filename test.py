import torch
from config import config
from functions.visualization import visualization_with_zoom
from misc.data_loader import DataLoader
from misc.unet import UNet
from skimage.metrics import structural_similarity   as ssim
from skimage.metrics import mean_squared_error      as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

index = 1366

CONSTRAINT = True
INITRECON  = True

model_name = f'data_prox_network_constraint_{str(CONSTRAINT)}_init_regul_{str(INITRECON)}'

D = DataLoader(index, CONSTRAINT, INITRECON)
X, Y = D[index]

model = UNet(1, 1, CONSTRAINT).to(config.device)
model.load_state_dict(torch.load('models/' + model_name))
model.eval()
model.to(config.device)

Y = torch.Tensor(Y).to(config.device)
Y = Y.unsqueeze(0)

out, res = model(Y)
out = out.cpu().detach().numpy().squeeze()
out = out.astype('float32')
res = res.cpu().detach().numpy().squeeze()
res = res.astype('float32')

print(f'MSE:  {mse(X.flatten(), out.flatten())}')
print(f'SSIM: {ssim(X.flatten(), out.flatten(), data_range=X.max() - X.min())}')
print(f'PSNR: {psnr(X.flatten(), out.flatten())}')

    
visualization_with_zoom(out.T, True, f'results/{model_name}.pdf')
