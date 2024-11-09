from dataclasses import dataclass
from typing import Tuple, List

# We want this to be immutable so that it can't be
# accidentally re-used and modified in a way that
# causes bugs.
@dataclass(frozen=True)
class EEGConfig:
    dataset: object
    eeg_dim = 512
    subject_range: Tuple[int, int]
    session_range: Tuple[int, int]
    filter_range: Tuple[int, int] = (5, 95)
    channel_range: Tuple[str, str] = ('Fp1', 'PO10')
    
    def __post_init__(self):
        # We use this over simple assignment because this is an
        # immutable object.
        object.__setattr__(self, 'num_channels',
                           self.channel_range[1] - self.channel_range[0]
                           )
        
    def get_range(self, type: str) -> List[int]:
        if type == 'subjects':
            return range(self.subject_range[0], self.subject_range[1] + 1)
        elif type == 'sessions':
            return range(self.session_range[0], self.session_range[1] + 1)
        else:
            raise ValueError(f"Invalid type: {type}")