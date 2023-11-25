import torch 
import torch.nn as nn
import torch.nn.functional as F
class recurrent_module(nn.Module):
    """Recurrent part of model. Currently made of 2 stacked LSTM bidirectional layers.
        output shape: (batch_size, seq_len, hidden_size)
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=False):
        super(recurrent_module, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size//2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size,self.hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output
class convolutional_module(nn.Module):
    """Convolutional part of model. Currently made of 2 stacked 1D convolutional layers.
        output shape: (batch_size, hidden_size, resolution)
    """
    def __init__(self, input_size, hidden_size, kernel_size=3,dropout=0.2):
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
        output = nn.functional.avg_pool1d(x_bar, kernel_size=self.kernel_size)
        return output
class self_supervised_model(nn.Module):
    """Combines recurrent and convolutional modules and adds a fully connected layer.
        output shape: (batch_size,sequence_length, 1) so that is likelihood of expert at each time step.
        I think that is something closer to how we would like the model to work in practice but, our data does not super support it. 
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True, kernel_size=3):
        super(self_supervised_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.kernel_size = kernel_size
        self.recurrent_module = recurrent_module(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.convolutional_module = convolutional_module(input_size, hidden_size, kernel_size, dropout)
        self.fc = nn.Linear(hidden_size, 1) ## binary task
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        recurrent_output = self.recurrent_module(x)
        convolutional_output = self.convolutional_module(x)
        output = torch.cat((recurrent_output, convolutional_output.permute(0, 2, 1)), dim=1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output