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
    eeg_dim = 512
    num_channels: int
    target_num_channels = 128
    expansion_factor = 1
    
    # These are tuples containing the high/low
    # or start/end of a range of values.
    subject_range: Tuple[int, int] = (1, 1)
    filter_range: Tuple[int, int] = (5, 95)
    channel_range: Tuple[str, str]
    
    def __post_init__(self):
        # We cannot simply assign because object is immutable.
        object.__setattr__(self, 'expansion_factor',
                           self.target_num_channels // self.num_channels
                           )
        
    def get_subject_range(self):
        return range(self.subject_range[0], self.subject_range[1] + 1)