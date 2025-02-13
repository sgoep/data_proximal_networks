import torch
import numpy as np

class config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_size = 128
    ran_angles = np.pi/3
    n_angles   = 120
    
    delta = 0.03
    alpha = 0.1

    len_train  = 500
    len_test   = 100
    num_epochs = 50
    batch_size = 6
    shuffle = True
    num_workers = 2
   
    lr = 1e-4
 
    
