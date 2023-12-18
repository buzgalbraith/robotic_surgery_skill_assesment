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
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=False, num_states=10, states_type:str = 'alternating'):
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
            start_idx = (N - self.num_states) // 2
            end_idx = start_idx + self.num_states
            return output[:, start_idx:end_idx, :]
        elif self.states_type == 'alternating':
            N = output.shape[1]
            indices = torch.linspace(0, N - 1, self.num_states).long()
            return torch.index_select(output, dim=1, index=indices)

    
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.get_states(output)
        return output

class convolutional_module(nn.Module):
    """Convolutional part of model. Currently made of 2 stacked 1D convolutional layers.
        output shape: (batch_size, hidden_size, resolution)
    """
    def __init__(self, input_size, hidden_size, kernel_size=3,dropout=0.2, resolution=5, skip=True):
        super(convolutional_module, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resolution = resolution
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(in_channels=self.input_size, out_channels=64, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=self.hidden_size, kernel_size=1)
        self.skip = skip
        if self.skip:
              self.skip_layer = torch.nn.Conv1d(in_channels=self.input_size, out_channels=self.hidden_size, kernel_size=3, stride=1)
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x_perm = x.permute(0, 2, 1)
        x_bar = self.conv_1(x_perm)
        x_bar = self.relu(x_bar)
        x_bar = self.dropout1(x_bar)
        x_bar = self.batch_norm1(x_bar)
        x_bar = self.conv_2(x_bar)
        if self.skip:
            x_perm_down = self.skip_layer(x_perm)
            x_bar = x_bar + x_perm_down
        x_bar = self.relu(x_bar)
        x_bar = self.dropout2(x_bar)
        x_bar = self.batch_norm2(x_bar)
        x_fin = nn.AdaptiveAvgPool1d(self.resolution)(x_bar)
        x_fin_perm = x_fin.permute(0, 2, 1)#.reshape(1,-1, self.hidden_size)
        return x_fin_perm
        
class Base_Model(nn.Module):
    """Combines recurrent and convolutional modules and adds a fully connected layer.
        output shape: (batch_size,learned features, 1) so that is likelihood of expert at each time step.
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True, kernel_size=3, num_states=10, resolution=2, verbose=False, states_type='alternating'):
        super(Base_Model, self).__init__()
        self.verbose = verbose
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.states_type =states_type
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.num_states = num_states
        self.recurrent_module = recurrent_module(input_size, hidden_size, num_layers, dropout, bidirectional, num_states = self.num_states, states_type=self.states_type)
        self.convolutional_module = convolutional_module(input_size, hidden_size, kernel_size, dropout, resolution =self.resolution)
        self.fc = nn.Linear((self.num_states+self.resolution)*self.hidden_size , 1) ## binary task
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        recurrent_output = self.recurrent_module(x)
        convolutional_output = self.convolutional_module(x)
        output = torch.cat((recurrent_output, convolutional_output), dim=1)
        output = self.fc(output.reshape(1, -1))
        output = self.sigmoid(output)
        return output