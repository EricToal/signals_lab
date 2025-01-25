from .eeg_config import EEGConfig
from .eeg_processor import EEGProcessor

from torch.utils.data import Dataset
import torch
from typing import List
from itertools import chain
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

class CombinedProcessor(Dataset):
    '''
    This class combines multiple EEGProcessor objects into a single dataset.
    
    Args:
        configs: A list of EEGConfig objects containing settings for the processors.
        
    Attributes:
        processors: A list of EEGProcessor objects.
        segment_iterator: An iterator that yields segments from the processor generators.
    '''
    def __init__(self, configs: List[EEGConfig], debug: bool = False):
        self.logger = logging.getLogger(__name__)
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        self.logger.debug('Creating CombinedProcessor and populating fields.')

        self.processors = [EEGProcessor(config) for config in configs]
        self.segment_iterator = self._create_segment_iterator()
        self.dir = './'
        self.filename = 'complete_dataset_tensor.pth'
        self.expected_shape = (configs[0].num_channels, configs[0].eeg_dim)
        
    def stack_data(self):
        '''
        This method stacks all processed EEG data into a single tensor.
        
        Returns: A tensor containing all processed EEG data.
        '''
        self.logger.debug('Stacking all processed EEG data.')
        return torch.vstack(list(self.segment_iterator))
    
    def save_data(self, path):
        '''
        This method saves all processed EEG data to a file.
        
        Args:
            path: The file path to save the stacked tensors to.
        '''
        path = path if path else self.dir + self.filename
        self.logger.debug(f'Saving data to {path}.')
        torch.save(self.stack_data(), path)
        
    def _create_segment_iterator(self):
        '''
        This method creates an iterator which manages the generators from the processor objects.
        
        Returns: An iterator that yields segments from the processor generators.
        '''
        self.logger.debug('Creating segment iterator.')
        all_generators = (processor.processed_data_gen() for processor in self.processors)
        return chain.from_iterable(all_generators)
    
    def __getitem__(self, idx):
        '''
        Ignores the index and returns the next segment from the processor
        generators until they are exhausted.
        
        Returns: The next segment from the processor generators.
        '''
        self.logger.debug(f'Getting next segment')
        try:
            return next(self.segment_iterator)
        except StopIteration:
            self.logger.debug('No more data to process.')
            return torch.zeros(self.expected_shape, dtype=torch.float32)
    
    def __len__(self):
        '''
        Returns: The total number of segments in the dataset.
        '''
        return sum(processor.length for processor in self.processors)