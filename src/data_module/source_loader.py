from data_module.time_series_dataset import TimeSeriesDataset
import os
import pandas as pd 
import torch
DATA_PATH = './data/processed_data'
class Source_Loader(TimeSeriesDataset):
    def __init__(self, trials = None, verbose = False, mode="binary", batch_size=1):
        super().__init__(trials, verbose, mode, batch_size)
        self.dataset = "JIGSAW"
        self.save_dir = DATA_PATH + "/" + self.dataset + "/"
        self.trials = trials
        self.batch_size = batch_size
        self.tasks = ['Suturing', 'Knot_Tying', 'Needle_Passing']
        self.load_data()
    def load_target_data(self):
            scores_location = self.save_dir + "METADATA/"
            files = [scores_location + f for f in os.listdir(scores_location) if f.endswith('.csv')]
            all_targets = []
            all_trials = []
            for file in files:
                df = pd.read_csv(file)
                if self.trials is not None:
                    df = df[df['task'].str.contains('|'.join(self.trials))]
                all_trials += df['task'].values.tolist()
                df = df[['robotic_surgery_experience']]
                if self.mode == "binary":
                    df = df.applymap(lambda x: 1 if x == "E" else 0)
                else:
                    df = df.applymap(lambda x: 2 if x == "E" else 1 if x == "I" else 0)
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
            task = trail[:-5]
            self.file_list.append(self.save_dir + task + "/" +trail + ".csv")
        self.input_data = self.load_input_data(self.dataset)