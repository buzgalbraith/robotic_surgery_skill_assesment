from data_module.self_supervised_dataset import Self_Supervised_Dataset
from model_module.base_model import Base_Model
from model_module.supervised_model import Supervised_Model
from model_module.self_supervised_model import Self_Supervised_Model
from model_module.utils import plot_loss_history, plot_confidence_interval

import torch

## set globals 
INPUT_SIZE = 40
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
HIDDEN_SIZE = 2**4
KERNEL_SIZE = 3
NUM_STATES = 9
RESOLUTION = 2
BATCH_SIZE = 5 ## was 35
TOTAL_EPOCHS = 10
ALPHA = 0.2
LAMBDA = 0.1
T = 10

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    self_supervised_dataset = Self_Supervised_Dataset(ROMSA_trials = ['X01'], JIGSAW_trials = ['B005', 'D002'], mode="binary", verbose=False, batch_size = BATCH_SIZE)
    self_supervised_model = Self_Supervised_Model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE, num_states = NUM_STATES, resolution = RESOLUTION).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(self_supervised_model.parameters(), lr=ALPHA, weight_decay=LAMBDA)
    for epoch in range(TOTAL_EPOCHS):
        self_supervised_model.self_supervised_train_step(self_supervised_dataset= self_supervised_dataset, criterion=criterion, optimizer=optimizer, device = device, epoch =epoch, total_epochs=TOTAL_EPOCHS, T=T)