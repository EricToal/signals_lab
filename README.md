# Current State

Able to glue MOABB datasets together by chaining generators.

Combined_processor is compatible with pytorch dataloader and random_split.

## Usage
Look in dataset_integration_test.ipynb for a usage example.

## Config
Each dataset needs a config object like shown below. For all params look in eeg_config.py.

config1 = EEGConfig(
    dataset=BI2015b(),
    num_channels=32,
    subject_range=(1, 5),
    filter_range=(5, 95),
    channel_range=("Fp1", "PO10"),
    debug=True
)