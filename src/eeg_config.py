from dataclasses import dataclass
from typing import Tuple

# We want this to be immutable so that it can't be
# accidentally re-used and modified in a way that
# causes bugs.
@dataclass(frozen=True)
class EEGConfig:
    '''
    This class contains configuration settings for processing EEG data
    Using the EEGProcessor class and its Combined form.
    '''
    dataset: object
    channel_range: Tuple[str, str]
    num_channels: int
    sample_rate: int
    sample_duration: int
    eeg_dim: int = 512
    subject_range: Tuple[int, int] = (1, 1)
    filter_range: Tuple[int, int] = (5, 95)
    
        
    def get_subject_range(self):
        return range(self.subject_range[0], self.subject_range[1] + 1)