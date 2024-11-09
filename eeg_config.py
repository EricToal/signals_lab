from dataclasses import dataclass
from typing import Tuple, List

# We want this to be immutable so that it can't be
# accidentally re-used and modified in a way that
# causes bugs.
@dataclass(frozen=True)
class EEGConfig:
    dataset: object
    eeg_dim: 512
    num_channels: int
    target_num_channels: 128
    subject_range: Tuple[int, int]
    session_range: Tuple[int, int]
    filter_range: Tuple[int, int] = (5, 95)
    channel_range: Tuple[str, str]
        
    def get_subject_range(self):
        return range(self.subject_range[0], self.subject_range[1] + 1)