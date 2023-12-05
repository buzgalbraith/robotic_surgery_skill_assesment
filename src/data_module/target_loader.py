from data_module.time_series_dataset import TimeSeriesDataset
import os
import pandas as pd
import torch
DATA_PATH = './data/processed_data'
class Target_Loader(TimeSeriesDataset):
    def __init__(self, trials = None, verbose = False, mode="binary", batch_size=1):
        super().__init__(trials, verbose, mode, batch_size)
        self.trials = trials
        self.dataset = "ROMSA"
        self.save_dir = DATA_PATH + "/" + self.dataset + "/"
        self.batch_size = batch_size
        self.tasks = ["Pea_on_a_Peg", 'Post_and_Sleeve', 'Wire_Chaser']
        self.load_data()
    def load_target_data(self):
        scores_location = self.save_dir + "METADATA/"
        files = [scores_location + f for f in os.listdir(scores_location) if f.endswith('.csv')]
        all_targets = []
        all_trials = []
        for file in files:
            if self.verbose:
                print("reading in {0}".format(file))
            df = pd.read_csv(file)
            if self.trials is not None:
                df = df[df['File_name'].str.contains('|'.join(self.trials))]
            all_trials += df['File_name'].values.tolist()
            df = df[['Score']]
            all_targets += df.values.tolist()
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
    def load_data(self):
        self.target_data = self.load_target_data()
        for trail in self.trails:
            task = trail[4:-3]
            if 'Wire_Chaser'  in task:
                task = 'Wire_Chaser'
            self.file_list.append(self.save_dir + task + "/" + trail + ".csv")
        self.input_data = self.load_input_data(self.dataset)