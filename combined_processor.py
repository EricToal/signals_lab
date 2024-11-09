from eeg_config import EEGConfig
from eeg_processor import EEGProcessor

from torch.utils.data import Dataset
import torch
from typing import List
from itertools import chain

class CombinedProcessor(Dataset):
    '''
    This class combines multiple EEGProcessor objects into a single dataset.
    
    Args:
        configs: A list of EEGConfig objects containing settings for the processors.
        
    Attributes:
        processors: A list of EEGProcessor objects.
        segment_iterator: An iterator that yields segments from the processor generators.
    '''
    def __init__(self, configs: List[EEGConfig]):
        self.processors = [EEGProcessor(config) for config in configs]
        self.segment_iterator = self._create_segment_iterator()
        self.dir = './'
        self.filename = 'complete_dataset_tensor.pth'
        
    def stack_data(self):
        '''
        This method stacks all processed EEG data into a single tensor.
        
        Returns: A tensor containing all processed EEG data.
        '''
        return torch.vstack(list(self.segment_iterator))
    
    def save_data(self, path):
        '''
        This method saves all processed EEG data to a file.
        
        Args:
            path: The file path to save the stacked tensors to.
        '''
        path = path if path else self.dir + self.filename
        torch.save(self.stack_data(), path)
        
    def _create_segment_iterator(self):
        '''
        This method creates an iterator which manages the generators from the processor objects.
        
        Returns: An iterator that yields segments from the processor generators.
        '''
        all_generators = (processor.process_data() for processor in self.processors)
        return chain.from_iterable(all_generators)
    
    def __getitem__(self, idx):
        '''
        Ignores the index and returns the next segment from the processor
        generators until they are exhausted.
        
        Returns: The next segment from the processor generators.
        '''
        try:
            return next(self.segment_iterator)
        except StopIteration:
            raise IndexError("Dataset is exhausted.")
    
    def __len__(self):
        '''
        Returns: The total number of segments in the dataset.
        '''
        # We use a nested sum to count the number of segments in each processor.
        # We use the sum method since we are using a generator.
        return sum(sum(1 for _ in processor.process_data()) for processor in self.processors)
