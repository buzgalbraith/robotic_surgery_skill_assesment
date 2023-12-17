from model_module.base_model import Base_Model
from data_module.time_series_dataset import TimeSeriesDataset
import time
import os
import torch
import glob
class Supervised_Model(Base_Model):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True, kernel_size=3, num_states=10, resolution=2, verbose=False):
        super(Supervised_Model, self).__init__(input_size, hidden_size, num_layers, dropout, bidirectional, kernel_size, num_states, resolution,verbose)
    def save_model(self, path ='saved_models/supervised_model/'):
        os.makedirs(path, exist_ok=True)
        save_path = path +str(time.time()) + "_model.pt"
        torch.save(self.state_dict(), save_path)
    def load_model(self, path=None):
        """loads last saved model if path is none."""
        if path is None:
            path = max(glob.glob("saved_models/supervised_model/*.pt"), key=os.path.getctime)
            if self.verbose:
                print("loading model from {}".format(path))
        else:
            print_path = path.split('/')[-1]
            print("Loading model from path: ", print_path)
        self.load_state_dict(torch.load(path))
    def train_step(self, dataset, optimizer, criterion, device, epoch, total_epochs):
        self.train()
        dataset.shuffle() ## shuffle dataset
        total_steps = len(dataset)
        epoch_loss = 0
        start_time = time.time()
        for i, (input_bath, target_bath) in enumerate(dataset):
            batch_loss = 0
            optimizer.zero_grad()
            for x,y in zip(input_bath, target_bath):
                x = x.to(device)
                y = y.to(device).squeeze()
                final_output = self(x).squeeze()
                loss = criterion(final_output, y)
                batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if self.verbose:
                print("batch [{}/{}] epoch [{}/{}] Loss: {:.4f}".format(i, total_steps, epoch, total_epochs, loss.item()))
            epoch_loss += loss.item()
        end_time = time.time()
        total_time = round(end_time-start_time, 2)
        if self.verbose:
            print("epoch [{}/{}] total_time: {} seconds".format(epoch,total_epochs, total_time))
        return epoch_loss
