from eeg_config import EEGConfig
import numpy as np

class EEGProcessor:
    def __init__(self, config: EEGConfig):
        self.config = config
        self.data = self.config.dataset.get_data(subjects = self.config.get_range('subjects'))
    
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
        # Stub for filtering data
        pass

    def _extract_channels(self, filtered_data):
        # Stub for extracting channels
        pass

    def _reshape_data(self, extracted_channels):
        # Stub for reshaping data
        pass

    def _process_segment(self, segment):
        # Stub for processing segment
        pass