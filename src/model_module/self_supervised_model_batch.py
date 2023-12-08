from model_module.base_model import Base_Model
from data_module.time_series_dataset import TimeSeriesDataset
import time
import os
import torch
import glob
class Self_Supervised_Model(Base_Model):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True, kernel_size=3, num_states=10, resolution=2, alpha=0.5, verbose=False):
        super(Self_Supervised_Model, self).__init__(input_size, hidden_size, num_layers, dropout, bidirectional, kernel_size, num_states, resolution)
        self.dropout_layer = torch.nn.Dropout(p=self.dropout, inplace=True)
        self.verbose = verbose
        self.alpha = alpha ## this weights the importance of pseudo labels vs source labels
    def save_model(self, path ='saved_models/self_supervised_Model/'):
        os.makedirs(path, exist_ok=True)
        save_path = path +str(time.time()) + "_model.pt"
        torch.save(self.state_dict(), save_path)
    def load_model(self, path=None):
        """loads last saved model if path is none."""
        if path is None:
            path = max(glob.glob("saved_models/supervised_model/*.pt"), key=os.path.getctime)
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
    def uncertainty_estimate(self, x, T=1000, get_ci=False):
        """uncertainty estimate for self supervised learning."""
        estimates = []
        for t in range(T):
            estimates.append(self.stochastic_forward(x))
        estimates = torch.stack(estimates)
        var = torch.var(estimates, dim=0)
        mean = torch.mean(estimates, dim=0)
        confidence_interval = None
        if get_ci:
            q = torch.tensor([0.025, 0.975])
            confidence_interval = torch.quantile(estimates, q, dim=0)
        return mean, var, confidence_interval
    def get_pseudo_labels(self, target_dataset, T=1000, test_steps=None):
        """get pseudo labels for target dataset."""
        pseudo_labels = []
        test_steps = len(target_dataset) if test_steps is None else test_steps
        for i, (x, _) in enumerate(target_dataset):
            if i > test_steps:
                break
            mean, var, confidence_interval = self.uncertainty_estimate(x, T=T)
            pseudo_labels.append(mean)
        pseudo_labels = torch.cat((pseudo_labels), dim=0)
        indices = torch.argsort(torch.abs(pseudo_labels - 0.5),dim=0, descending=False)
        pseudo_labels  = torch.index_select(pseudo_labels, dim=0, index=indices.flatten())
        return pseudo_labels, indices.flatten()




    def update_datasets(self, self_supervised_dataset, pseudo_labels,indices, K=5):
        """need to think of more ways to validate that this is working"""
        ## get indices of top K pseudo labels
        top_k_indices = indices[:K].flatten() ## get the top k indices
        input_data_to_add = []
        target_data_to_add = []
        batched_indices = dict()
        batch_size = len(self_supervised_dataset.target_dataset.input_data[0])
        K = min(K, len(pseudo_labels)) ## in cases where we have less than k pseudo labels available
        for k in range(K):
            batch_index, remainder = self_supervised_dataset.convert_to_batch(top_k_indices[k])
            input_data_to_add.append(self_supervised_dataset.target_dataset.input_data[batch_index][remainder])
            if batch_index not in batched_indices:
                batched_indices[batch_index] = [remainder]
            else:
                batched_indices[batch_index].append(remainder)
            target_data_to_add.append( 0.0 if pseudo_labels[k] < 0.5 else 1.0)
        ## make the input_data to add batched
        batch_input = []
        batch_label = []
        ## add to train dataset
        for i in range(len(input_data_to_add)):
            x = input_data_to_add[i]
            y = target_data_to_add[i]
            batch_input.append(x)
            batch_label.append(y)
            if len(batch_input) == batch_size:
                self_supervised_dataset.train_dataset[1].input_data += [batch_input]
                batch_label = torch.tensor(batch_label).reshape(-1,1)
                self_supervised_dataset.pseudo_data.target_data.append(batch_label)
                batch_input = []
                batch_label = []
        ## get the last possibly incomplete batch
        if len(batch_input) > 0:
            self_supervised_dataset.train_dataset[1].input_data += [batch_input]
            batch_label = torch.tensor(batch_label).reshape(-1,1)
            self_supervised_dataset.pseudo_data.target_data.append(batch_label)
        ## remove from target 
        for batch_index in batched_indices.keys():
            remainders = batched_indices[batch_index]
            num_removed = 0
            for remainder in sorted(remainders, reverse=True):
                del self_supervised_dataset.target_dataset.input_data[batch_index][remainder]
                del self_supervised_dataset.target_dataset.target_data[batch_index][remainder]
        return self_supervised_dataset
    def source_loss(self, source_dataset, optimizer, criterion, device,epoch, total_epochs):
        """loss function for source dataset."""
        total_steps = len(source_dataset)
        start_time = time.time()
        for i, (input_bath, target_bath) in enumerate(source_dataset):
            epoch_loss = 0
            optimizer.zero_grad()
            for x,y in zip(input_bath, target_bath):
                x = x.to(device)
                y = y.to(device).squeeze()
                final_output = self(x).squeeze()
                loss = criterion(final_output, y)
            loss.backward()
            optimizer.step()
            if self.verbose:
                print("Source batch [{}/{}] epoch [{}/{}] Loss: {:.4f}".format(i, total_steps, epoch, total_epochs, loss.item()))
            epoch_loss += loss.item()
        end_time = time.time()
        total_time = round(end_time-start_time, 2)
        if self.verbose:
            print("epoch [{}/{}] total_time: {} seconds".format(epoch,total_epochs, total_time))
        return epoch_loss
    def target_loss(self, target_dataset, optimizer, criterion, device, epoch, total_epochs):
        """loss function for target dataset."""
        if self.verbose:
            print("number of trials in target dataset during target ste: {}".format(target_dataset.get_num_trials()))
        total_steps = len(target_dataset)
        start_time = time.time()
        for i, (input_bath, target_bath) in enumerate(target_dataset):
            epoch_loss = 0
            optimizer.zero_grad()
            for x,y in zip(input_bath, target_bath):
                x = x.to(device)
                y = y.to(device).squeeze()
                final_output = self(x).squeeze()
                loss = criterion(final_output, y) 
            loss = loss * self.alpha
            loss.backward()
            optimizer.step()
            if self.verbose:
                print("Target batch [{}/{}] epoch [{}/{}] Loss: {:.4f}".format(i, total_steps, epoch, total_epochs, loss.item))
            epoch_loss += loss.item()
        end_time = time.time()
        total_time = round(end_time-start_time, 2)
        if self.verbose:
            print("epoch [{}/{}] total_time: {} seconds".format(epoch,total_epochs, total_time))
        return epoch_loss
    def self_supervised_loss(self, train_dataset, criterion):
        """loss function for self supervised learning.
            note that here we assume that there is also l2 regularization in the optimizer (weight decay)
        """
        ## this is going to need to figure out where the source and target data is and way the loss accordingly.
        return source_loss + self.alpha * target_loss
    def model_step(self, dataset, optimizer,criterion, device, epoch, total_epochs):
        self.train()
        ## shuffle datasets
        dataset.train_dataset[0].shuffle()
        dataset.train_dataset[1].shuffle()
        epoch_start = time.time()
        if self.verbose:
            print("number of trials in source train dataset: {}".format(dataset.train_dataset[0].get_num_trials()))
            print("number of trials in target train dataset: {}".format(dataset.train_dataset[1].get_num_trials()))
            print("number of trials in original target dataset: {}".format(dataset.target_dataset.get_num_trials()))
        source_loss = self.source_loss(dataset.train_dataset[0], optimizer, criterion, device,epoch, total_epochs)
        target_loss = self.target_loss(dataset.train_dataset[1], optimizer, criterion, device,epoch, total_epochs)

        epoch_finish = time.time()
        epoch_time = round(epoch_finish - epoch_start, 2)
        # epoch_loss = source_loss + source_loss
        return source_loss, target_loss
    def self_supervised_train_step(self, self_supervised_dataset, optimizer,criterion, device, epoch, total_epochs, T=1000, K=5):
        """self supervised training step."""
        ## get pseudo labels and model uncertainty
        pseudo_labels, indices = self.get_pseudo_labels(self_supervised_dataset.target_dataset, T=T)
        ## update datasets
        self_supervised_dataset = self.update_datasets(self_supervised_dataset, pseudo_labels,indices, K=K)
        ## train on updated source dataset.
        source_loss, target_loss = self.model_step(self_supervised_dataset, optimizer, criterion, device, epoch, total_epochs)
        return source_loss, target_loss

