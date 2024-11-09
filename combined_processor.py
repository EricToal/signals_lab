from eeg_config import EEGConfig
from eeg_processor import EEGProcessor

from torch.utils.data import Dataset
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
