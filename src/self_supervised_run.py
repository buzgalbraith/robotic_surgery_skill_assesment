from data_module.data_loader import TimeSeriesDataset
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
NUM_STATES = 10
RESOLUTION = 2
BATCH_SIZE = 5 ## was 35
TOTAL_EPOCHS = 10

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # dataset = TimeSeriesDataset('JIGSAW', trials = ['B005', 'D002'], mode="binary", verbose=False, batch_size = BATCH_SIZE) 
    dataset = TimeSeriesDataset('JIGSAW',  mode="binary", verbose=False, batch_size = BATCH_SIZE) 
    self_supervised_model = Self_Supervised_Model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE, num_states = NUM_STATES, resolution = RESOLUTION).to(device)
    self_supervised_model.load_model()
    for x, y in dataset:
        plot_confidence_interval(self_supervised_model, x, y)
        break
        # stochastic_forward = self_supervised_model.stochastic_forward(x)
        # print(len(stochastic_forward))
        # mean, var, ci = self_supervised_model.uncertainty_estimate(x)
        # print(mean)
        # print(var)
        # print(mean.shape)
        # print(var.shape)
        # print(ci)
        # print(ci.shape)
        # break