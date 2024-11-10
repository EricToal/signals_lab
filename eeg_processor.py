from eeg_config import EEGConfig
import numpy as np
import torch
import logging

logging=logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

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
        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Creating EEGProcessor and populating fields.')
        
        self.config = config
        self.data = self.config.dataset.get_data(subjects = [i for i in self.config.get_subject_range()])
    
    def process_data(self):
        '''
        This method is the main entry point for processing EEG data.
        
        Returns: A generator that yields processed EEG data.
        '''
        for subject_id, sessions in self.data.items():
            # Allow for early stopping so that we can limit range of subjects.
            if subject_id not in range(self.config.subject_range[0], self.config.subject_range[1] + 1):
                break

            self.logger.debug(f'Processing subject {subject_id}.')

            for session_id, runs in sessions.items():
                self.logger.debug(f'Processing session {session_id}.')

                for run_id, raw_data in runs.items():
                    self.logger.debug(f'Processing run {run_id}.')

                    filtered_data = self._filter_data(raw_data)
                    extracted_channels = self._extract_channels(filtered_data)
                    reshaped_data = self._reshape_data(extracted_channels)
                    
                    for segment in reshaped_data:
                        yield self._process_segment(segment)
                        
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
        
        return extracted_channels[:num_chunks * eeg_dim] \
            .reshape(-1, eeg_dim, self.config.num_channels)

    def _process_segment(self, segment):
        '''
        This method processes a single EEG data segment.
        
        Args:
            segment: The EEG data segment to process. This should have been
                     segmented and reshaped by the _reshape_data method.
                     
        Returns:
            A tensor containing the processed EEG data segment.
            '''
        self.logger.debug('Processing segment.')
        return torch.Tensor(segment).repeat(1, self.config.expansion_factor).unsqueeze(0)