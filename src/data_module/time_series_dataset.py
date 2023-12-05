
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random
DATA_PATH = './data/processed_data'
"""Class for loading time series dataset"""
## probably refactor this as source and target dataset classes. 
class TimeSeriesDataset(Dataset):
    def __init__(self, trials = None, verbose = False, mode="binary", batch_size=1):
        self.verbose = verbose
        self.mode = mode
        self.batch_size = batch_size
        self.file_list = []
        self.trials = trials
        self.input_data = []
        self.target_data = []
    def shuffle(self):
        zipped = list(zip(self.input_data, self.target_data))
        random.shuffle(zipped)
        self.input_data, self.target_data = zip(*zipped)
        # why is this a tuple?
        self.input_data = list(self.input_data)
        self.target_data = list(self.target_data)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
    def load_input_data(self,dataset ,normalize = True):
        data = []
        path = DATA_PATH + "/" + dataset + "/" 
        scaler = StandardScaler()
        for file_path in self.file_list:
            if self.verbose:
                print("reading in {0}".format(file_path))
            df = pd.read_csv(file_path)
            if normalize:
                df = scaler.fit_transform(df)
            time_series = torch.tensor(df, dtype=torch.float32)
            time_series = time_series.reshape(1, time_series.shape[0], time_series.shape[1])
            data.append(time_series)
        batch_data = []
        all_data = []
        for time_series in data:
            batch_data.append(time_series)
            if len(batch_data) == self.batch_size:
                all_data.append(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            all_data.append(batch_data)
        return all_data
    def get_num_trials(self):
        # return (len(self.input_data)-1) * self.batch_size + len(self.input_data[-1])
        total_trails = 0
        for trial in self.input_data:
            total_trails += len(trial)
        return total_trails 
if __name__ == "__main__":
    pass 