from .eeg_config import EEGConfig
from .eeg_processor import EEGProcessor

from torch.utils.data import Dataset
import torch
from typing import List
from itertools import chain
import logging
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.signal import stft

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

class CombinedProcessor(Dataset):
    '''
    This class combines multiple EEGProcessor objects into a single dataset.
    
    Args:
        configs: A list of EEGConfig objects containing settings for the processors.
        
    Attributes:
        processors: A list of EEGProcessor objects.
        segment_iterator: An iterator that yields segments from the processor generators.
    '''
    def __init__(self, configs: List[EEGConfig], nperseg: int, noverlap: int, window="hann"):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Creating CombinedProcessor and populating fields.')

        self.processors = [EEGProcessor(config) for config in configs]
        self.segment_iterator = self._create_segment_iterator()
        self.dir = './'
        self.filename = 'complete_dataset_tensor.pth'
        self.expected_shape = (configs[0].num_channels, configs[0].eeg_dim)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.max_spectr_channels = max(config.num_channels for config in configs)
        
    def stack_data(self):
        '''
        This method stacks all processed EEG data into a single tensor.
        
        Returns: A tensor containing all processed EEG data.
        '''
        self.logger.debug('Stacking all processed EEG data.')
        return torch.vstack(list(self.segment_iterator))
    
    def save_data(self, path):
        '''
        This method saves all processed EEG data to a file.
        
        Args:
            path: The file path to save the stacked tensors to.
        '''
        path = path if path else self.dir + self.filename
        self.logger.debug(f'Saving data to {path}.')
        torch.save(self.stack_data(), path)

    def compute_spectrograms(self, segment: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute spectrograms matching notebook's axis order.
        Returns:
            spectrograms: [Channels, Freq, Time] (no channel dim)
        """
        num_channels = segment.shape[0]
        spectrograms = []
        
        for channel in range(num_channels):
            f, t, Zxx = stft(segment, sample_rate/2, nperseg=self.nperseg, noverlap=self.noverlap, window=self.window)
            Zxx_truncated = Zxx[:128, :-1]
            magnitude = np.abs(Zxx_truncated)
            db_scale = librosa.amplitude_to_db(magnitude, ref=np.max)
            db_scale = db_scale.T
            db_scale = np.squeeze(db_scale)
            
            spectrograms.append(db_scale) 
        
        return np.stack(spectrograms, axis=0) 

    def compare_spectrograms(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """
        Compute MSE between original and reconstructed spectrograms.
        Args:
            original: Spectrogram of shape [Channels, Freq, Time].
            reconstructed: Spectrogram of shape [Channels, Freq, Time].
        Returns:
            mse: Mean squared error for each channel.
        """
        num_channels = original.shape[0]
        mse = []
    
        for channel in range(num_channels):
            orig_flat = original[channel].flatten()
            rec_flat = reconstructed[channel].flatten()
    
            mse.append(np.mean((orig_flat - rec_flat) ** 2))
    
        return np.array(mse)

    def plot_spectrograms(self, original: np.ndarray, reconstructed: np.ndarray, mse: np.ndarray, sample_rate: int, requested_channels: int = 1):
        """
        Simplified plotting method to match the notebook's approach.
        
        Args:
            original: Original spectrograms [Channels, Freq, Time]
            reconstructed: Reconstructed spectrograms [Channels, Freq, Time]
            mse: MSE per channel [Channels]
            sample_rate: Sample rate of the audio
            requested_channels: Number of channels to visualize
        """
    
        # Plot spectrogram comparison
        fig, axes = plt.subplots(requested_channels, 2, figsize=(15, 3 * requested_channels))
        
        if requested_channels == 1:
            axes = axes[np.newaxis, :]  # Maintain 2D axes array
    
        for ch in range(requested_channels):
            original_squeezed = np.squeeze(original[ch]) 
            reconstructed_squeezed = np.squeeze(reconstructed[ch])
    
            # Original spectrogram
            librosa.display.specshow(
                original_squeezed, 
                sr=sample_rate // 2, 
                hop_length=self.noverlap, 
                x_axis='time', 
                y_axis='hz', 
                ax=axes[ch, 0]
            )
            axes[ch, 0].set_title(f'Original (Ch {ch+1})')
            
            # Reconstructed spectrogram
            librosa.display.specshow(
                reconstructed_squeezed, 
                sr=sample_rate // 2, 
                hop_length=self.noverlap, 
                x_axis='time', 
                y_axis='hz', 
                ax=axes[ch, 1]
            )
            axes[ch, 1].set_title(f'Reconstructed (Ch {ch+1})')
    
        plt.tight_layout()
        plt.show()
        
    def _create_segment_iterator(self):
        '''
        This method creates an iterator which manages the generators from the processor objects.
        
        Returns: An iterator that yields segments from the processor generators.
        '''
        self.logger.debug('Creating segment iterator.')
        all_generators = (processor.processed_data_gen() for processor in self.processors)
        return chain.from_iterable(all_generators)
    
    def __getitem__(self, idx):
        '''
        Ignores the index and returns the next segment from the processor
        generators until they are exhausted.
        
        Returns: The next segment from the processor generators.
        '''
        self.logger.debug(f'Getting next segment')
        try:
            return next(self.segment_iterator)
        except StopIteration:
            self.logger.debug('No more data to process.')
            return torch.zeros(self.expected_shape, dtype=torch.float32)
    
    def __len__(self):
        '''
        Returns: The total number of segments in the dataset.
        '''
        return sum(processor.length for processor in self.processors)