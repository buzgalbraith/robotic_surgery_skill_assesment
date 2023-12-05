
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
    def __init__(self, dataset, trials = None, verbose = False, mode="binary", batch_size=1):
        self.verbose = verbose
        self.dataset = dataset
        self.mode = mode
        self.batch_size = batch_size
        self.file_list = []
        self.trials = trials
        self.save_dir = DATA_PATH + "/" + dataset + "/"
        if self.dataset == 'JIGSAW':
            self.tasks = ['Suturing', 'Knot_Tying', 'Needle_Passing']
        else:
            self.tasks = ["Pea_on_a_Peg", 'Post_and_Sleeve', 'Wire_Chaser']
        self.target_data = self._load_target_data()
        # self.trails = set(trail[:4] for trail in self.trails)
        if self.dataset == 'JIGSAW':
            for trail in self.trails:
                task = trail[:-5]
                self.file_list.append(self.save_dir + task + "/" +trail + ".csv")
        else:
            for trail in self.trails:
                ## task is after the 4th char before the 7th to last
                task = trail[4:-3]
                if 'Wire_Chaser'  in task:
                    task = 'Wire_Chaser'
                self.file_list.append(self.save_dir + task + "/" + trail + ".csv")
        self.input_data = self._load_input_data()
        if self.verbose:
            print("input data: ", len(self.input_data))
            for data in self.input_data:
                print(len(data))
            print("target data: ", len(self.target_data))
            for data in self.target_data:
                print(len(data))
    def load_target_data(self):
        scores_location = self.save_dir + "METADATA/"
        files = [scores_location + f for f in os.listdir(scores_location) if f.endswith('.csv')]
        all_targets = []
        all_trials = []
        for file in files:
            if self.verbose:
                print("reading in {0}".format(file))
            df = pd.read_csv(file)
            if self.dataset == 'JIGSAW':
                if self.trials is not None:
                    df = df[df['task'].str.contains('|'.join(self.trials))]
                all_trials += df['task'].values.tolist()
                df = df[['robotic_surgery_experience']]
                if self.mode == "binary":
                    df = df.applymap(lambda x: 1 if x == "E" else 0)
                else:
                    df = df.applymap(lambda x: 2 if x == "E" else 1 if x == "I" else 0)
            else:
                if self.trials is not None:
                    df = df[df['File_name'].str.contains('|'.join(self.trials))]
                all_trials += df['File_name'].values.tolist()
                df = df[['Score']]
            all_targets += df.values.tolist()
        # all_targets = torch.tensor(all_targets, dtype=torch.float32)
        ## now want to batch the targets
        labels = []
        batch_labels = []
        for label in all_targets:
            batch_labels.append(torch.tensor(label, dtype=torch.float32))
            if len(batch_labels) == self.batch_size:
                labels.append(batch_labels)
                batch_labels = []
        if len(batch_labels) > 0:
            labels.append(batch_labels)
        self.trails = all_trials
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
    def shuffle(self):
        zipped = list(zip(self.input_data, self.target_data))
        random.shuffle(zipped)
        self.input_data, self.target_data = zip(*zipped)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
    def get_num_trials(self):
        # return (len(self.input_data)-1) * self.batch_size + len(self.input_data[-1])
        total_trails = 0
        for trial in self.input_data:
            try:
                total_trails += len(trial)
            except:
                print(trial)
        return total_trails 

class Self_Supervised_Loader(TimeSeriesDataset):
    def __init__(self, ROMSA_trials = None, JIGSAW_trials = None,  verbose = False, mode="binary", batch_size=1):
        super().__init__(dataset, trials, verbose, mode, batch_size)
        self.ROMSA_trials = ROMSA_trials
        self.JIGSAW_trials = JIGSAW_trials
        self.batch_size = batch_size
        self.source_dataset = TimeSeriesDataset('JIGSAW', trials = self.JIGSAW_trials, mode="binary", verbose=False, batch_size = self.batch_size)
        self.target_dataset = TimeSeriesDataset('ROMSA', trials = self.ROMSA_trials, mode="binary", verbose=False, batch_size = self.batch_size)
        self.pseudo_data = []
        self.train_dataset = [self.source_dataset, self.pseudo_data]
if __name__ == "__main__":
    pass 