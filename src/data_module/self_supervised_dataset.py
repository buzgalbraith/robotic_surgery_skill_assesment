from src.data_module.source_loader import Source_Loader
from src.data_module.target_loader import Target_Loader
from src.data_module.time_series_dataset import TimeSeriesDataset   
## object which can hold both the source and target dataset
class Self_Supervised_Dataset():
    def __init__(self, ROMSA_trials = None, JIGSAW_trials = None,  verbose = False, mode="binary", batch_size=1):
        self.ROMSA_trials = ROMSA_trials
        self.JIGSAW_trials = JIGSAW_trials
        self.batch_size = batch_size
        self.source_dataset = Source_Loader(trials = self.JIGSAW_trials, mode=mode, verbose=False, batch_size = self.batch_size)
        self.target_dataset = Target_Loader(trials = self.ROMSA_trials, mode=mode, verbose=False, batch_size = self.batch_size)
        self.pseudo_data = TimeSeriesDataset(trials = self.ROMSA_trials, mode=mode, verbose=False, batch_size = self.batch_size)
        self.train_dataset = [self.source_dataset, self.pseudo_data]
    def get_source_batchnumbers(self):
        """counts the number of trials left in each batch of the target dataset"""
        batch_out = [len(batch) for batch in self.target_dataset.input_data]
        return batch_out 
    def convert_to_batch(self, index):
        batch_numbers = self.get_source_batchnumbers()
        batch_index = 0
        remainder = index
        for i, batch_number in enumerate(batch_numbers):
            if remainder < batch_number:
                batch_index = i
                break
            remainder -= batch_number
        return batch_index, remainder