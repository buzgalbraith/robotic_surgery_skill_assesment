
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
DATA_PATH = './data/processed_data'
"""Class for loading time series dataset"""
class TimeSeriesDataset(Dataset):
    def __init__(self, dataset, trials = None, verbose = False):
        self.verbose = verbose
        self.dataset = dataset
        if self.dataset == 'JIGSAW':
            self.tasks = ['Suturing', 'Knot_Tying', 'Needle_Passing']
        else:
            self.tasks = ['Pea_on_a_Peg']
        self.file_list = []
        self.trials = trials
        self.save_dir = DATA_PATH + "/" + dataset + "/"
        for task in self.tasks: 
            task_path = self.save_dir + task + "/"
            files = [task_path + f for f in os.listdir(task_path) if f.endswith('.csv')] 
            if self.trials is not None:
                files = [f for f in files if f[-8:-4] in self.trials]
            self.file_list += files
        self.input_data = self._load_input_data()
        self.target_data = self._load_target_data()
        if self.verbose:
            print("input data: ", len(self.input_data))
            print("target data: ", len(self.target_data))

    def _load_target_data(self):
        scores_location = self.save_dir + "METADATA/"
        files = [scores_location + f for f in os.listdir(scores_location) if f.endswith('.csv')]
        labels = []
        for file in files:
            if self.verbose:
                print("reading in {0}".format(file))
            df = pd.read_csv(file)
            if self.dataset == 'JIGSAW':
                if self.trials is not None:
                    df = df[df['task'].str.contains('|'.join(self.trials))]
                df = df[['overall_score']]
            else:
                if self.trials is not None:
                    df = df[df['trial'].isin(self.trials)]
                df = df[['score']]
            labels += df.values.tolist()
        labels = torch.tensor(labels, dtype=torch.float32)
        return labels

    def _load_input_data(self, normalize = True):
        data = []
        path = DATA_PATH + "/" + self.dataset + "/" 
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
        return data


    def __len__(self):
        return len(self.input_data)



    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]



if __name__ == "__main__":
    pass 