import torch

class config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_size = 128
    ran_angles = torch.pi/3
    n_angles   = 120
    
    delta = 0.05
    alpha = 0.1

    len_train  = 1
    len_test   = 1
    num_epochs = 2
    batch_size = 4
    shuffle = True
    num_workers = 4
   
    lr = 1e-4
 
    