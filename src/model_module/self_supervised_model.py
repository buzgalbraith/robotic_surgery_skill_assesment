from model_module.base_model import Base_Model
from data_module.data_loader import TimeSeriesDataset
import time
import os
import torch
import glob
class Self_Supervised_Model(Base_Model):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True, kernel_size=3, num_states=10, resolution=2):
        super(Self_Supervised_Model, self).__init__(input_size, hidden_size, num_layers, dropout, bidirectional, kernel_size, num_states, resolution)
        self.dropout_layer = torch.nn.Dropout(p=self.dropout, inplace=True)
    def save_model(self, path ='saved_models/self_supervised_Model/'):
        os.makedirs(path, exist_ok=True)
        save_path = path +str(time.time()) + "_model.pt"
        torch.save(self.state_dict(), save_path)
    def load_model(self, path=None):
        """loads last saved model if path is none."""
        if path is None:
            path = max(glob.glob("saved_models/supervised_model/*.pt"), key=os.path.getctime)
            print("loading model from {}".format(path))
        self.load_state_dict(torch.load(path))
    def stochastic_forward(self,batch):
        """stochastic forward pass for self supervised learning."""
        self.train() ## set to train for dropout
        batch_output = torch.zeros((len(batch), 1))
        with torch.no_grad():
            for i, x in enumerate(batch):
                drop_x = self.dropout_layer(x)
                batch_output[i] = self(drop_x)
        return batch_output
    def uncertainty_estimate(self, x, T=1000):
        """uncertainty estimate for self supervised learning."""
        estimates = []
        for t in range(T):
            estimates.append(self.stochastic_forward(x))
        estimates = torch.stack(estimates)
        var = torch.var(estimates, dim=0)
        mean = torch.mean(estimates, dim=0)
        q = torch.tensor([0.025, 0.975])
        confidence_interval = torch.quantile(estimates, q, dim=0)
        return mean, var, confidence_interval
