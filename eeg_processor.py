from eeg_config import EEGConfig
import numpy as np
from pandas import DataFrame
from typing import Generator

class EEGProcessor:
    def __init__(self, config: EEGConfig):
        self.config = config
        self.data = self.config.dataset.get_data(subjects = self.config.get_range('subjects'))
        