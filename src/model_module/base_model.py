import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import glob
import os
class recurrent_module(nn.Module):
    """Recurrent part of model. Currently made of 2 stacked LSTM bidirectional layers.
        output shape: (batch_size, states_kept, hidden_size)
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=False, num_states=10, states_type:str = 'last'):
        super(recurrent_module, self).__init__()
        assert states_type in ['last', 'first', 'last_and_first', 'middle', 'alternating']
        self.input_size = input_size
        self.hidden_size = hidden_size//2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_states = num_states
        self.lstm = nn.LSTM(input_size,self.hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.states_type = states_type
    def get_states(self, output):
        if self.states_type == 'last':
            return output[:,-self.num_states:,:]  ##outputs the last n states
        elif self.states_type == 'first':  
            return output[:,:self.num_states,:] ## outputs the first n states
        elif self.states_type == 'last_and_first':
            return torch.cat((output[:,:self.num_states//2,:], output[:,-self.num_states//2:,:]), dim=1) ## outputs the first and last n//2 states
        elif self.states_type == 'middle':
            N = output.shape[1]
            return output[:,N//2-self.num_states//2:N//2+self.num_states//2,:] ## outputs the middle n states
        elif self.states_type == 'alternating':
            ## want to get every N//n state
            N = output.shape[1]
            return output[:,::N//self.num_states,:] ## outputs the middle n states

    
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.get_states(output)
        return output.reshape(1 , self.num_states,-1)
class convolutional_module(nn.Module):
    """Convolutional part of model. Currently made of 2 stacked 1D convolutional layers.
        output shape: (batch_size, hidden_size, resolution)
    """
    def __init__(self, input_size, hidden_size, kernel_size=3,dropout=0.2, resolution=5):
        super(convolutional_module, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, self.hidden_size, kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.resolution = resolution
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_bar = self.conv1(x)
        x_bar = self.relu(x_bar)
        x_bar = self.dropout1(x_bar)
        x_bar = self.batch_norm1(x_bar)
        x_bar = self.conv2(x_bar)
        x_bar = self.relu(x_bar)
        x_bar = self.dropout2(x_bar)
        x_bar = self.batch_norm2(x_bar)
        ## want it to always output to 5
        pool_kernel_size = x_bar.shape[2]//self.resolution
        output = nn.functional.avg_pool1d(x_bar, kernel_size=pool_kernel_size)
        return output
class Base_Model(nn.Module):
    """Combines recurrent and convolutional modules and adds a fully connected layer.
        output shape: (batch_size,learned features, 1) so that is likelihood of expert at each time step.
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True, kernel_size=3, num_states=10, resolution=2, verbose=False):
        super(Base_Model, self).__init__()
        self.verbose = verbose
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.num_states = num_states
        self.recurrent_module = recurrent_module(input_size, hidden_size, num_layers, dropout, bidirectional, num_states = self.num_states)
        self.convolutional_module = convolutional_module(input_size, hidden_size, kernel_size, dropout, resolution =self.resolution)
        self.fc = nn.Linear(resolution*hidden_size + num_states * hidden_size, 1) ## binary task
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        recurrent_output = self.recurrent_module(x)
        convolutional_output = self.convolutional_module(x)
        output = torch.cat((recurrent_output, convolutional_output.permute(0, 2, 1)), dim=1)
        output = self.fc(output.reshape(1, -1))
        output = self.sigmoid(output)
        return output