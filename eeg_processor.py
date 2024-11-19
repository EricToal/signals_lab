from eeg_config import EEGConfig
from itertools import tee
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')

class EEGProcessor:
    '''
    This class processes EEG data using an EEGConfig object.
    
    Args:
        config: An EEGConfig object containing configuration settings.
        expansion_factor: An integer specifying the expansion factor for the EEG data.
        
    Attributes:
        config: An EEGConfig object containing configuration settings.
        data: A dictionary containing EEG data for each subject, session, and run.
    '''
    def __init__(self, config: EEGConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.length = self.gen_length(self.processed_data_gen())
    
    def processed_data_gen(self):
        '''
        This method is the main entry point for processing EEG data.
        
        Returns: A generator that yields processed EEG data.
        '''
        for subject_id, sessions in next(self._subject_gen()).items():
            self.logger.debug(f'Processing subject {subject_id}.')

            for session_id, runs in sessions.items():
                self.logger.debug(f'Processing session {session_id}.')

                for run_id, raw_data in runs.items():
                    self.logger.debug(f'Processing run {run_id}.')

                    filtered_data = self._filter_data(raw_data)
                    extracted_channels = self._extract_channels(filtered_data)
                    #reshaped_data = self._reshape_data(extracted_channels)
                    
                    for segment in self._process_segment(extracted_channels):
                        yield segment
    
    def gen_length(self, gen):
        '''
        This method calculates the length of a generator.
        
        Args:
            gen: The generator to calculate the length of.
            
        Returns: The length of the generator.
        '''
        self.logger.debug('Calculating generator length.')
        gen, gen_copy = tee(gen)
        length = sum(1 for _ in gen_copy)
        return length
    
    def _subject_gen(self):
        for subject in self.config.get_subject_range():
            yield self.config.dataset.get_data(subjects=[subject])
                        
    def _filter_data(self, raw_data):
        '''
        This method filters raw EEG data.
        
        Args:
            raw_data: The raw EEG data to filter.
            
        Returns: 
            Filtered EEG data as a float32 pandas DF.
        '''
        self.logger.debug(f'Filtering data from {self.config.filter_range[0]} to {self.config.filter_range[1]} Hz')
        return raw_data \
            .filter(self.config.filter_range[0], self.config.filter_range[1], verbose=0) \
            .to_data_frame() \
            .astype(np.float32)

    def _extract_channels(self, filtered_data):
        '''
        This method extracts specificc channel columns from filtered EEG data.
        
        Args:
            filtered_data: The filtered EEG data to extract columns from.
            
        Returns:
            A numpy array containing the extracted channel columns.
        '''
        self.logger.debug(f'Extracting channels {self.config.channel_range[0]} to {self.config.channel_range[1]}.')
        return filtered_data \
            .loc[:, self.config.channel_range[0]:self.config.channel_range[1]] \
            .values

    def _reshape_data(self, extracted_channels):
        '''
        This method segments and reshapes extracted EEG data.
        
        Args:
            extracted_channels: The extracted EEG data to segment and reshape.
            
        Returns:
            A reshaped numpy array containing segmented EEG data.
        '''
        eeg_dim = self.config.eeg_dim
        self.logger.debug(f'Reshaping data into segments of {eeg_dim} samples.')
        num_chunks = extracted_channels.shape[0] // self.config.eeg_dim
        self.logger.debug(f'Extracted {num_chunks} segments.')
        
        return extracted_channels[:num_chunks * eeg_dim] \
            .reshape(self.config.num_channels, eeg_dim)

    def _process_segment(self, extracted_channels):
        
        eeg_dim = self.config.eeg_dim
        num_channels = self.config.num_channels
        self.logger.debug(f'Reshaping data into segments of {eeg_dim} samples.')
        
        total_elements = extracted_channels.shape[0]
        self.logger.debug(f'Total elements in extracted channels: {total_elements}.')
        
        num_segments = total_elements // eeg_dim
        self.logger.debug(f'Extracted {num_segments} complete segments.')
        
        for i in range(num_segments):
            start = i * eeg_dim
            end = start + eeg_dim
            segment = extracted_channels[start:end]
            yield segment.reshape(num_channels, eeg_dim)
        
    
    def __len__(self):
        return self.length