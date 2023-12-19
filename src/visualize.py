from src.data_module.target_loader import Target_Loader
from src.data_module.utils import plot_ROMSA_progression, plot_ROMSA_progression_and_model_estimate
from src.model_module.self_supervised_model import Self_Supervised_Model
from src.model_module.utils import plot_confidence_interval_unsupervised
### 
ROMSA_PATH = 'data/processed_data/ROMSA'
INPUT_SIZE = 40
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
MODE = 'binary'
## we an also play with the model parameters more
HIDDEN_SIZE = 2**4
KERNEL_SIZE = 3
NUM_STATES = 2
RESOLUTION = 1
BATCH_SIZE = 3 ## was 35
TOTAL_EPOCHS = 3
ALPHA = 0.5
AVG = True
LAMBDA = 0.1
T = 10000
K = 3
VERBOSE = False
###
if __name__ == "__main__":
    print("starting ")
    # test_task = 'Post_and_Sleeve'
    # test_task = 'Pea_on_a_Peg'
    test_task = "Wire_Chaser"
    test_participant = "X01"
    device = "cpu"
    plot_ROMSA_progression(task=test_task, participant=test_participant)
    participant_data = Target_Loader(trials=[test_participant], tasks=[test_task], batch_size=BATCH_SIZE)
    labels = participant_data.file_list
    model = Self_Supervised_Model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE, num_states = NUM_STATES, resolution = RESOLUTION).to(device)
    model.load_model(mode = "validate")
    participant_data.unbatch()
    plot_ROMSA_progression_and_model_estimate(model=model, input=participant_data.input_data, task=test_task, participant=test_participant, T=T)

