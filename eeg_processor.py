from eeg_config import EEGConfig
import numpy as np
import torch

class EEGProcessor:
    def __init__(self, config: EEGConfig, expansion_factor: int = 1):
        self.config = config
        self.data = self.config.dataset.get_data(subjects = self.config.get_subject_range())
    
    def process_data(self):
        '''
        This method is the main entry point for processing EEG data.
        
        Returns: A generator that yields processed EEG data.
        '''
        for subject_id, sessions in self.data.items():
            for session_id, runs in sessions.items():
                for run_id, raw_data in runs.items():
                    filtered_data = self._filter_data(raw_data)
                    extracted_channels = self._extract_channels(filtered_data)
                    reshaped_data = self._reshape_data(extracted_channels)
                    
                    for segment in reshaped_data:
                        yield self._process_segment(segment)
                        
    def _filter_data(self, raw_data):
        return raw_data \
            .filter(self.config.filter_range[0], self.config.filter_range[1], verbose=0) \
            .to_data_frame() \
            .astype(np.float32)

    def _extract_channels(self, filtered_data):
        return filtered_data \
            .loc[:, self.config.channel_range[0]:self.config.channel_range[1]] \
            .values

    def _reshape_data(self, extracted_channels):
        eeg_dim = self.config.eeg_dim
        num_chunks = extracted_channels.shape[0] // self.config.eeg_dim
        
        return extracted_channels[:num_chunks * eeg_dim] \
            .reshape(-1, eeg_dim, self.config.num_channels)

    def _process_segment(self, segment):
        eeg = torch.Tensor(segment).repeat(1, self.expansion_factor)