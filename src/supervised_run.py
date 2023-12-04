from data_module.data_loader import TimeSeriesDataset
from model_module.base_model import Base_Model
from model_module.supervised_model import Supervised_Model
from model_module.utils import plot_loss_history
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
    supervised_model = Supervised_Model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE, num_states = NUM_STATES, resolution = RESOLUTION).to(device)
    epoch = 0 
    loss_history = []
    while epoch < TOTAL_EPOCHS:
        optimizer = torch.optim.Adam(supervised_model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        epoch_loss = supervised_model.train_step(dataset, optimizer, criterion, device, epoch, TOTAL_EPOCHS)
        print("epoch [{}/{}] epoch_loss: {:.4f}".format(epoch, TOTAL_EPOCHS, epoch_loss))
        epoch += 1
        loss_history.append(epoch_loss)
    plot_loss_history(loss_history, "loss history")    

        