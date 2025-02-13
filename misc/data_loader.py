# %%
import numpy as np
import torch


class DataLoader(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ID, CONSTRAINT, INITRECON):
        'Initialization'
        self.ID         = ID
        self.CONSTRAINT = CONSTRAINT
        self.INITRECON  = INITRECON
      
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ID)
    
    def __getitem__(self, index):
        'Generates one sample of data'          
        
        X = np.load(f'data/phantom/phantom_{str(index)}.npy')
        if self.INITRECON:      
            Y = np.load(f'data/init_regul/init_regul_{str(index)}.npy')        
        else:
            Y = np.load(f'data/fbp/fbp_{str(index)}.npy')
        
        X = X[None,:,:]
        Y = Y[None,:,:]
        
        return X, Y
    

# data_loader = DataLoader([i for i in range(100)], False, False)
# next(iter(data_loader))[0].shape
# %%