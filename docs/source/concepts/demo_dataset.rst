
.. _loading_dataset:

Loading a Dataset Object
========================

Dataset objects are the core of the Experanto library. They extend single experiments by adding functionality that allows them to be used directly as **dataloaders** for machine learning tasks.

Key Features of Dataset Objects
-------------------------------
Dataset objects provide several essential features:

- **Sampling Rate**: Defines the frequency of equally spaced interpolation times across the entire experiment. This ensures consistency in temporal data alignment.
- **Chunk Size**: Determines the number of values returned when calling the ``__getitem__`` method. This is crucial for deep learning models utilizing **3D convolutions over time**, as single elements or small chunk sizes would be insufficient for meaningful temporal patterns.
- **Modality Configuration**: Specifies the details of the interpolation, including:

  - The **interpolation method** used.
  - **Conditions** that the data must fulfill.
  - **Transforms** applied to the data (e.g., normalization, resizing, cropping, greyscale conversion).

Loading a Dataset
-----------------
To load a dataset, follow the steps below:

.. code-block:: python

    import sys
    from experanto.datasets import ChunkDataset
    from torch.utils.data import DataLoader
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    # Define root folder containing experiment data
    root_folder = '../data/allen_data'
    sampling_rate = 8  # Global sampling rate
    chunk_size = 32  # Chunk size for dataset loading

    # Define modality configuration for training set (screen and response interpolation)
    train_dataset = ChunkDataset(
        root_folder=f'{root_folder}/experiment_951980471',
        global_sampling_rate=sampling_rate,
        global_chunk_size=chunk_size,
        modality_config={
            'screen': {
                'sampling_rate': None,
                'chunk_size': None,
                'valid_condition': {
                    'tier': 'train',
                    'stim_type': 'stimulus.Frame',  # Include both images and videos
                    'stim_type': 'stimulus.Clip'
                },
                'offset': 0,
                'sample_stride': 4,
                # Necessary for the Allen dataset to handle blank spaces after stimuli
                'include_blanks': True,
                'transforms': {
                    'Normalize': {
                        '_target_': 'torchvision.transforms.Normalize',
                        'mean': 80.0,
                        'std': 60.0
                    },
                    'Resize': {
                        '_target_': 'torchvision.transforms.Resize',
                        'size': [144, 256]
                    },
                    'CenterCrop': {
                        '_target_': 'torchvision.transforms.CenterCrop',
                        'size': 144
                    },
                    'greyscale': True  # Convert to greyscale data
                },
                'interpolation': {}
            },
            'responses': {
                'sampling_rate': None,
                'chunk_size': None,
                'offset': 0.1,
                'transforms': {
                    'standardize': True
                },
                'interpolation': {
                    'interpolation_mode': 'nearest_neighbor'
                }
            },
        }
    )

This configuration ensures that:

- **Screen data** is preprocessed with normalization, resizing, cropping, and greyscale conversion.
- **Response data** undergoes standardization and nearest-neighbor interpolation.

Other modalities can be defined in the same manner as **Responses**.

Sampling Data from the Dataset
------------------------------
We can confirm the creation and functionality of our datasets by sampling some data.
To sample data from the dataset, we can simply index into it. For example, to sample the first data chunk:

.. code-block:: python

    # Interpolation showcase using the dataset object
    sample = train_dataset[0]

    # Print the keys and their respective shapes
    print(sample.keys())
    for key in sample.keys():
        print(f'This is shape {sample[key].shape} for modality {key}')

This will output something like:

.. code-block:: text

    dict_keys(['screen', 'responses', 'timestamps'])
    This is shape torch.Size([1, 32, 144, 144]) for modality screen
    This is shape torch.Size([32, 12]) for modality responses
    This is shape torch.Size([32, 1]) for modality timestamps

Defining DataLoaders
---------------------
Once the dataset is verified, we can define **DataLoader** objects for training or other purposes. This allows easy batch processing during training:

.. code-block:: python

    # Define a DataLoader for the training set
    data_loader['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True)

Verifying DataLoader Functionality
----------------------------------
To confirm that the **DataLoader** works as expected, we can iterate over it and inspect the batch data. For example, to check the shapes of the data in each batch:

.. code-block:: python

    # Interpolation showcase using the data_loaders
    for batch_idx, batch_data in enumerate(data_loaders['train']):
        # batch_data is a dictionary with keys 'screen', 'responses', and 'timestamps'
        screen_data = batch_data['screen']
        responses = batch_data['responses']
        timestamps = batch_data['timestamps']
        
        # Print or inspect the batch
        print(f"Batch {batch_idx}:")
        print("Screen Data:", screen_data.shape)
        print("Responses:", responses.shape)
        print("Timestamps:", timestamps.shape)
        break

This will output something like:

.. code-block:: text

    Batch 0:
    Screen Data: torch.Size([15, 1, 32, 144, 144])
    Responses: torch.Size([15, 32, 12])
    Timestamps: torch.Size([15, 32, 1])
