import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from misc.unet import UNet
from misc.data_loader import DataLoader
from config import config


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/(3**2 *m.in_channels)))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/ m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            

def TRAIN(model, TrainDataGenerator, TestDataGenerator, error, optimizer, device, NumEpochs):
    
    torch.cuda.empty_cache()
    # gc.collect()
    
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    train_loss_list = np.zeros(NumEpochs)
    val_loss_list = np.zeros(NumEpochs)
        
    for epoch in np.arange(1, NumEpochs+1):
        train_loss = 0.0
        val_loss = 0.0
        
        model.train()
        for X, Y in TrainDataGenerator:
            optimizer.zero_grad()
            
            X = X.to(device=device, dtype=torch.float)
            Y = Y.to(device=device, dtype=torch.float)
                        
            output, _  = model(Y)

            loss    = error(output.double(), X.double())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.item()
            history['loss'].append(train_loss)
        
        model.eval()
        counter = 0
        for X, Y in TrainDataGenerator:
            
            X = X.to(device=device, dtype=torch.float)
            Y = Y.to(device=device, dtype=torch.float)
            
            output, _  = model(Y)
        
            loss    = error(output.double(), X.double())
            
            val_loss += loss.data.item()
            history['val_loss'].append(val_loss)
        
        train_loss = train_loss/config.len_train
        val_loss   = val_loss/config.len_test
        
        print(f'Epoch {epoch}/{NumEpochs}, Train Loss: {train_loss:.8f} and Val Loss: {val_loss:.8f}')

        train_loss_list[epoch-1] = train_loss
        val_loss_list[epoch-1]   = val_loss      
     
    return history, train_loss_list, val_loss_list


def start_training(CONSTRAINT, INITRECON, null_space_network, model_name):
    
    model = UNet(1, 1, CONSTRAINT, null_space_network).to(config.device)
    
    torch.manual_seed(42)
    model.apply(init_weights)
     
    model_optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model_error     = nn.MSELoss()
    # 
    model_params    = {'batch_size': config.batch_size,
                       'shuffle': config.shuffle,
                       'num_workers': config.num_workers}
                       
    model_train_set = DataLoader([i for i in range(config.len_train)], CONSTRAINT, INITRECON)
    model_test_set  = DataLoader([i for i in range(config.len_train, config.len_train+config.len_test)], CONSTRAINT, INITRECON)
    
    TrainDataGen = torch.utils.data.DataLoader(model_train_set, **model_params)
    TestDataGen  = torch.utils.data.DataLoader(model_test_set,  **model_params)
    
    # model_name = f'data_prox_network_constraint_{str(CONSTRAINT)}_init_regul_{str(INITRECON)}'
    
    print('##################### Start training ######################### ')
    print('Model: ' + model_name) 
    print('Size training set: ' + str(config.len_train))
    print('Size test set: ' + str(config.len_test))
    print('Epochs: ' + str(config.num_epochs))
    print('Learning rate: ' + str(config.lr))
    print('Training on: ' + str(config.device))
    print('############################################################## ')
       
    history, train_loss_list, val_loss_list = TRAIN(model,
                                                    TrainDataGen,
                                                    TestDataGen,
                                                    model_error,
                                                    model_optimizer,
                                                    config.device,
                                                    config.num_epochs)
    
    
    loss_plot = 'loss_' + model_name + '.pdf'
    
    plt.figure()
    plt.plot(train_loss_list,  marker='x', label="Training Loss")
    plt.plot(val_loss_list,  marker='x', label="Validation Loss")
    plt.ylabel('loss_' + model_name, fontsize=22)
    plt.legend()
    plt.savefig(loss_plot)
    
    print('######################## Saving model ######################## ') 
    print('###### ' + model_name + ' ######') 
    torch.save(model.state_dict(), f'models/{model_name}')
    print('########################## Finished ########################## ') 
    

if __name__ == "__main__":

    # FBP + RES
    start_training(CONSTRAINT=False, INITRECON=False, null_space_network=False, name="FBP_RES")
    
    # TV + RES
    start_training(CONSTRAINT=False, INITRECON=True, null_space_network=False, name="TV_RES")
    
    # FBP + NSN
    start_training(CONSTRAINT=False, INITRECON=False, null_space_network=True, name="FBP_NSN")
    
    # TV + NSN
    start_training(CONSTRAINT=False, INITRECON=True, null_space_network=True, name="TV_NSN")

    # TV + DP
    start_training(CONSTRAINT=True, INITRECON=True, null_space_network=True, name="TV_DP")
    
