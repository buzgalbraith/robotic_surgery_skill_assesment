from src.data_module.self_supervised_dataset import Self_Supervised_Dataset
from src.model_module.base_model import Base_Model
from src.model_module.supervised_model import Supervised_Model
from src.model_module.self_supervised_model import Self_Supervised_Model
from src.model_module.utils import plot_target_and_source_loss


import torch
import tqdm
## set globals 
INPUT_SIZE = 40
NUM_LAYERS = 2
DROPOUT = 0.4
BIDIRECTIONAL = True
MODE = 'binary'
HIDDEN_SIZE = 2**4
KERNEL_SIZE = 3
NUM_STATES = 2
RESOLUTION = 1
BATCH_SIZE = 5 ## was have tried 3,5,10
TOTAL_EPOCHS = 50 ## have tried 10, 100, 250,1000
ALPHA = 0.75 ## have tried 0.2 ,0,7, 0.85
AVG = True
LAMBDA = 0.02## used to be 0.2
T = 250 ## have tried 10, 100, 250,1000"
STATES_TYPE = "last_and_first"


import tqdm 


K = BATCH_SIZE
VERBOSE = False

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # self_supervised_dataset = Self_Supervised_Dataset(ROMSA_trials = ['X01'], JIGSAW_trials = ['B005', 'D002'], mode="binary", verbose=False, batch_size = BATCH_SIZE)
    self_supervised_dataset = Self_Supervised_Dataset(mode=MODE, batch_size =BATCH_SIZE)
    self_supervised_model = Self_Supervised_Model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE, num_states = NUM_STATES, resolution = RESOLUTION, states_type=STATES_TYPE).to(device)
    criterion = torch.nn.BCELoss() 
    self_supervised_model.load_model()
    # optimizer = torch.optim.Adam(self_supervised_model.parameters(), lr=ALPHA, weight_decay=LAMBDA)
    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.001, weight_decay=0.0005)
    source_loss_history = []
    target_loss_history = []
    self_supervised_loss_history = []
    for epoch in tqdm.tqdm(range(TOTAL_EPOCHS)):
        source_loss, target_loss = self_supervised_model.self_supervised_train_step(self_supervised_dataset= self_supervised_dataset, criterion=criterion, optimizer=optimizer, device = device, epoch =epoch, total_epochs=TOTAL_EPOCHS, T=T, K=K)
        source_loss_history.append(source_loss)
        target_loss_history.append(target_loss)
        self_supervised_loss_history.append(source_loss + target_loss)
        # if epoch>1 and abs(target_loss_history[-1] -target_loss_history[-2])<0.0001: ## check how well this is working. we also want to add a learning rate sche
        #     break
    ## plot the loss history
    plot_target_and_source_loss(target_loss_history, source_loss_history, title="Target and source loss", save_fig=True)
    self_supervised_model.save_model()
