from data_module.data_loader import TimeSeriesDataset
from model_module.model import recurrent_module, convolutional_module, self_supervised_model
import torch

## set globals 
INPUT_SIZE = 40
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
HIDDEN_SIZE = 2**4
KERNEL_SIZE = 3

if __name__ == "__main__":
    dataset = TimeSeriesDataset('JIGSAW', trials = ['B001', 'B002'])



    recurrent_model = recurrent_module(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL)
    conv_model = convolutional_module(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, kernel_size= KERNEL_SIZE, dropout = DROPOUT)
    model = self_supervised_model(input_size= INPUT_SIZE,hidden_size=HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional =BIDIRECTIONAL, kernel_size= KERNEL_SIZE)
    for i, (x, y) in enumerate(dataset):
        print(x.shape)
        print(y.shape)
        rec_output = recurrent_model(x)
        conv_output = conv_model(x)
        final_output = model(x)
        print(rec_output.shape, "rec")
        print(conv_output.shape, "conv")
        print(final_output.shape, "final")
        print('-'*100)
        if i==4:
            break
        