from data_module.target_loader import Target_Loader
from data_module.utils import plot_ROMSA_progression
from model_module.self_supervised_model import Self_Supervised_Model
from model_module.utils import plot_confidence_interval_unsupervised
### 
ROMSA_PATH = 'data/processed_data/ROMSA'
INPUT_SIZE = 40
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
MODE = 'binary'
HIDDEN_SIZE = 2**4
KERNEL_SIZE = 3
NUM_STATES = 9
RESOLUTION = 2
BATCH_SIZE = 5 ## was 35
TOTAL_EPOCHS = 3
ALPHA = 0.5
AVG = True
LAMBDA = 0.1
T = 10
K = 3
VERBOSE = False
###
if __name__ == "__main__":
    test_task = 'Wire_Chaser'
    test_participant = "X09"
    device = "cpu"
    plot_ROMSA_progression(task=test_task, participant=test_participant)
    participant_data = Target_Loader(trials=[test_participant], tasks=[test_task], batch_size=BATCH_SIZE)
    labels = participant_data.file_list
    model = Self_Supervised_Model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE, num_states = NUM_STATES, resolution = RESOLUTION, verbose=VERBOSE, avg=AVG).to(device)
    model.load_model()
    participant_data.unbatch()
    plot_confidence_interval_unsupervised(model=model, x=participant_data.input_data, labels=labels, T=10, title=test_task+"_"+test_participant)
    
    # for i, x in enumerate(participant_data.input_data):
    #     used_labels = labels[start:end]
    #     plot_confidence_interval_unsupervised(model=model, x=x, labels=labels, T=10, it_num=i)
    #     start = end
    #     end += BATCH_SIZE

    
