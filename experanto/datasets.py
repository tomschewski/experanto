from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import os

import json
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor, Compose, Lambda
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import functools
import importlib

from .configs import DEFAULT_MODALITY_CONFIG
from .experiment import Experiment
from .interpolators import ImageTrial, VideoTrial
from .utils import add_behavior_as_channels, replace_nan_with_batch_mean
from .intervals import TimeInterval, find_intersection_between_two_interval_arrays, get_stats_for_valid_interval

# see .configs.py for the definition of DEFAULT_MODALITY_CONFIG
DEFAULT_MODALITY_CONFIG = dict()


class SimpleChunkedDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        sampling_rate: float,
        chunk_size: int,
        interp_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self._experiment = Experiment(
            root_folder,
            interp_config,
        )
        self.device_names = self._experiment.device_names
        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self._sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rate
        )
        self.DataPoint = namedtuple("DataPoint", self.device_names)

    def __len__(self):
        return int(len(self._sample_times) / self.chunk_size)

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        times = self._sample_times[s : s + self.chunk_size]
        data, _ = self._experiment.interpolate(times)

        # check if we use phaseshifts before using them
        if self._experiment.devices["responses"].use_phase_shifts:
            phase_shifts = self._experiment.devices["responses"]._phase_shifts
            timestamps_neurons = (times - times.min())[:, None] + phase_shifts[None, :]
            data["timestamps"] = timestamps_neurons

        else:
            data["timestamps"] = (times - times.min())[:, None]

        # Hack-2: add batch dimension for screen
        if len(data["screen"].shape) != 4:
            data["screen"] = data["screen"][:, None, ...]
        return data


class Mouse2pStaticImageDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        tier: str,
        offset: float,
        stim_duration: float,
        interp_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.offset = offset
        self.stim_duration = stim_duration
        self._experiment = Experiment(
            root_folder,
            interp_config,
        )
        self.device_names = self._experiment.device_names
        self.DataPoint = namedtuple("DataPoint", self.device_names)
        self._read_trials()

    def _read_trials(self):
        screen = self._experiment.devices["screen"]
        self._trials = [
            t
            for t in screen.trials
            if isinstance(t, ImageTrial) and t.get_meta("tier") == self.tier
        ]
        s_idx = np.array([t.first_frame_idx for t in self._trials])
        if len(s_idx):
            self._start_times = screen.timestamps[s_idx]
        else:
            self._start_times = np.array([])

    def __len__(self):
        return len(self._trials)

    def __getitem__(self, idx):
        assert isinstance(idx, int), "Index must be an integer"
        data = dict()
        for device_name, device in self._experiment.devices.items():
            if device_name == "screen":
                times = self._start_times[idx]
            else:
                Fs = device.sampling_rate
                times = (
                    self._start_times[idx]
                    + self.offset
                    + np.arange(0, self.stim_duration, 1.0 / Fs)
                )
            d, _ = device.interpolate(times)
            data[device_name] = d.mean(axis=0)
        return self.DataPoint(**data)


class Mouse2pVideoDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        tier: str,
        stim_duration: float,
        sampling_rate: float,
        subsample: bool,
        cut: bool,
        add_channel: bool,
        channel_pos: int,
        interp_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        """
        this dataloader returns the full the video resampled to the new freq rate
        subsampling frames ideally should be outside dataset but in the dataloader augmentations

        :param root_folder: path to the data folder
        :param tier: train/test/validation
        :param stim_duration: how many frames to take from the video
        :param sampling_rate: sampling rate to interpolate
        :param subsample: if we sample longer video from the non-start position
        :param cut: if we cut the video up to the stim_duration length or not
        :param add_channel: if video does not have channels, this flag shows if to add it
        :param channel_pos: if add_channel True and no channels are in the video, this would be the position to add it
        """
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.sampling_rate = sampling_rate
        self.stim_duration = stim_duration
        self._experiment = Experiment(
            root_folder,
            interp_config,
        )
        self.device_names = self._experiment.device_names
        # this is needed to match sensorium order only
        if "screen" in self.device_names and "responses" in self.device_names:
            start = ["screen", "responses"]
            self.device_names = tuple(start) + tuple(
                set(self.device_names).difference(set(start))
            )
        self.subsample = subsample
        self.cut = cut
        self.add_channel = add_channel
        self.channel_pos = channel_pos
        assert (
            0 <= channel_pos < 4
        ), "channels could be extended only for positions [0,3]"

        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self.DataPoint = namedtuple("DataPoint", self.device_names)
        self.MetaNeuro = namedtuple(
            "MetaNeuro", ["cell_motor_coordinates", "unit_ids", "fields"]
        )
        self._read_trials()

    def _read_trials(self):
        # reads all videos from valid tiers and saves times for them
        # also have saves the start and end time if test videos are in between
        # todo
        screen = self._experiment.devices["screen"]
        self._trials = [
            t
            for t in screen.trials
            if isinstance(t, VideoTrial) and t.get_meta("tier") == self.tier
        ]
        s_idx = np.array([t.first_frame_idx for t in self._trials])
        # todo - not sure if it should be t.first_frame_idx + t.num_frames
        e_idx = np.array([t.first_frame_idx + t.num_frames - 1 for t in self._trials])
        # todo - this uses the assumption that sampling_rate is less or equal the sampling rate of the screen stimuli
        if self.cut:
            assert all(
                [t.num_frames >= self.stim_duration for t in self._trials]
            ), "stim_duration should be smaller"
        if len(s_idx):
            self._start_times = screen.timestamps[s_idx]
            self._end_times = screen.timestamps[e_idx]
        else:
            self._start_times = np.array([])
            self._end_times = np.array([])

    def __len__(self):
        return len(self._trials)

    @property
    def neurons(self):
        loc_meta = {
            "cell_motor_coordinates": [],
            "unit_ids": [],
            "fields": [],
        }
        if "responses" in self._experiment.devices.keys():
            # todo - make it lazy loading? and read-only properties?
            root_folder = self._experiment.devices["responses"].root_folder
            meta = self._experiment.devices["responses"].load_meta()
            if "neuron_properties" in meta:
                cell_motor_coordinates = np.load(
                    root_folder / meta["neuron_properties"]["cell_motor_coordinates"]
                )
                unit_ids = np.load(root_folder / meta["neuron_properties"]["unit_ids"])
                fields = np.load(root_folder / meta["neuron_properties"]["fields"])

                loc_meta = {
                    "cell_motor_coordinates": cell_motor_coordinates,
                    "unit_ids": unit_ids,
                    "fields": fields,
                }

        return self.MetaNeuro(**loc_meta)

    def __getitem__(self, idx):
        """

        :param idx: idx of the video
        :return: this would return video in data['screen'] with shape of [t, h, w]
        """
        fs = self.sampling_rate
        times = np.arange(self._start_times[idx], self._end_times[idx], 1 / fs)
        # get all times possible
        # cut is needed
        if self.cut:
            if self.subsample:
                start = np.random.randint(0, len(times) - self.stim_duration)
                times = times[start : start + self.stim_duration]
            else:
                times = times[: self.stim_duration]

        data, _ = self._experiment.interpolate(times)

        if self.add_channel and len(data["screen"].shape) != 4:
            data["screen"] = np.expand_dims(data["screen"], axis=self.channel_pos)
        # this hack matches the shape for sensorium models
        if "responses" in data:
            data["responses"] = data["responses"].T
        return self.DataPoint(**data)


class ChunkDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        global_sampling_rate: None,
        global_chunk_size: None,
        add_behavior_as_channels: bool = False,
        replace_nans_with_means: bool = False,
        cache_data: bool = False,
        out_keys: Optional[Iterable] = None,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        seed: Optional[int] = None,
    ) -> None:
        """
        The full modality config is a nested dictionary.
        The following is an example of a modality config for a screen, responses, eye_tracker, and treadmill:

        screen:
          sampling_rate: null
          chunk_size: null
          valid_condition:
            tier: test
            stim_type: stimulus.Frame
          offset: 0
          sample_stride: 4
          include_blanks: false
          transforms:
            ToTensor:
              _target_: torchvision.transforms.ToTensor
            Normalize:
              _target_: torchvision.transforms.Normalize
              mean: 80.0
              std: 60.0
            Resize:
              _target_: torchvision.transforms.Resize
              size:
              - 144
              - 256
            CenterCrop:
              _target_: torchvision.transforms.CenterCrop
              size: 144
          interpolation: {}
        responses:
          sampling_rate: null
          chunk_size: null
          offset: 0.1
          transforms:
            standardize: true
          interpolation:
            interpolation_mode: nearest_neighbor
        eye_tracker:
          sampling_rate: null
          chunk_size: null
          offset: 0
          transforms:
            normalize: true
          interpolation:
            interpolation_mode: nearest_neighbor
        treadmill:
          sampling_rate: null
          chunk_size: null
          offset: 0
          transforms:
            normalize: true
          interpolation:
            interpolation_mode: nearest_neighbor
        """
        self.root_folder = Path(root_folder)
        self.data_key = self.get_data_key_from_root_folder(root_folder)

        self.modality_config = instantiate(modality_config)
        self.chunk_sizes, self.sampling_rates, self.chunk_s = {}, {}, {}
        for device_name in self.modality_config.keys():
            cfg = self.modality_config[device_name]
            self.chunk_sizes[device_name] = global_chunk_size or cfg.chunk_size
            self.sampling_rates[device_name] = global_sampling_rate or cfg.sampling_rate

        self.add_behavior_as_channels = add_behavior_as_channels
        self.replace_nans_with_means = replace_nans_with_means
        self.sample_stride = self.modality_config.screen.sample_stride
        self._experiment = Experiment(
            root_folder,
            modality_config,
            cache_data=cache_data,
        )
        self.device_names = self._experiment.device_names

        self.out_keys = out_keys or self.device_names
        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self._read_trials()
        self.initialize_statistics()
        
        self._screen_sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rates["screen"]
        )
        # iterate over the valid condition in modality_config["screen"]["valid_condition"] to get the indices of self._screen_sample_times that meet all criteria
        self._full_valid_sample_times_filtered = self.get_full_valid_sample_times(filter_for_valid_intervals=True)
        # self._full_valid_sample_times_unfiltered = self.get_full_valid_sample_times(filter_for_valid_intervals=False)

        # the _valid_screen_times are the indices from which the starting points for the chunks will be taken
        # sampling stride is used to reduce the number of starting points by the stride
        # default of stride is 1, so all starting points are used
        self._valid_screen_times = self._full_valid_sample_times_filtered[::self.sample_stride]

        self.transforms = self.initialize_transforms()

        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def _read_trials(self) -> None:
        screen = self._experiment.devices["screen"]
        self._trials = [t for t in screen.trials]
        start_idx = np.array([t.first_frame_idx for t in self._trials])
        self._start_times = screen.timestamps[start_idx]
        self._end_times = np.append(screen.timestamps[start_idx[1:]], np.inf)
        self.meta_conditions = {}
        for k in ["modality", "valid_trial"] + list(self.modality_config.screen.valid_condition.keys()):
            self.meta_conditions[k] = [t.get_meta(k) if t.get_meta(k) is not None else "blank" for t in self._trials]

    def initialize_statistics(self) -> None:
        """
        Initializes the statistics for each device based on the modality config.
        :return:
            instantiates self._statistics with the mean and std for each device
        """
        self._statistics = {}
        for device_name in self.device_names:
            self._statistics[device_name] = {}
            # If modality should be normalized, load respective statistics from file.
            if self.modality_config[device_name].transforms.get("normalization", False):
                mode = self.modality_config[device_name].transforms.normalization
                means = np.load(self._experiment.devices[device_name].root_folder / "meta/means.npy")
                stds = np.load(self._experiment.devices[device_name].root_folder / "meta/stds.npy")

                # if mode is a dict, it will override the means and stds
                if not isinstance(mode, str):
                    means = np.array(mode.get("means", means))
                    stds = np.array(mode.get("stds", stds))
                if mode == 'standardize':
                    # If modality should only be standarized, set means to 0.
                    means = np.zeros_like(means)
                elif mode == 'recompute_responses':
                     means = np.zeros_like(means)
                     stds = np.nanstd(self._experiment.devices["responses"]._data, 0)[None, ...]
                elif mode == 'recompute_behavior':
                     means = np.nanmean(self._experiment.devices[device_name]._data, 0)[None, ...]
                     stds = np.nanstd(self._experiment.devices[device_name]._data, 0)[None, ...]
                elif mode == 'screen_default':
                     means = np.array((80))
                     stds = np.array((60))

                self._statistics[device_name]["mean"] = means.reshape(1, -1)  # (n, 1) -> (1, n) for broadcasting in __get_item__
                self._statistics[device_name]["std"] = stds.reshape(1, -1)  # same as above

    # removed the dim transforms here since it is already handled in the interpolator
    def initialize_transforms(self):
        """
        Initializes the transforms for each device based on the modality config.
        :return:
        """
        transforms = {}
        for device_name in self.device_names:
            if device_name == "screen":
                transform_list = [v for v in self.modality_config.screen.transforms.values() if isinstance(v, torch.nn.Module)]

                # Apply channel reduction if specified in the config to lower dimensionalit for greyscale
                if self.modality_config.screen.transforms.get("greyscale", False):
                    transform_list.append(torchvision.transforms.Lambda(lambda x: x[0:1, :, :, :].unsqueeze(0)))               

            else:
                transform_list = [ToTensor()]

            # Normalization.
            if self.modality_config[device_name].transforms.get("normalization", False):
                transform_list.append(
                    torchvision.transforms.Normalize(self._statistics[device_name]["mean"], self._statistics[device_name]["std"])
                )

            transforms[device_name] = Compose(transform_list)
        return transforms
    
    def _get_callable_filter(self, filter_config):
        """
        Helper function to get a callable filter function from either a config or an already instantiated callable.
        Handles partial instantiation using hydra.utils.instantiate.
        
        Args:
            filter_config: Either a config dict/DictConfig or a callable function
            
        Returns:
            callable: The final filter function ready to be called with device_
        """
        # Check if it's already a callable (function)
        if callable(filter_config):
            # print(f"DEBUG: callable(filter_config) returned True for type {type(filter_config)}. Returning config directly.")
            return filter_config
        
        # Check if it's a config that needs instantiation
        if isinstance(filter_config, (dict, DictConfig)) and '__target__' in filter_config:
            try:
                # Manually handle instantiation for factory pattern with __partial__=True
                target_str = filter_config['__target__']
                module_path, func_name = target_str.rsplit('.', 1)
                
                # Import the module and get the factory function
                module = importlib.import_module(module_path)
                factory_func = getattr(module, func_name)
                
                # Prepare arguments for the factory function (excluding special keys)
                args = {k: v for k, v in filter_config.items() if k not in ('__target__', '__partial__')}
                
                # Call the factory function with its arguments to get the actual implementation function
                implementation_func = factory_func(**args)
                return implementation_func
                
            except (ImportError, AttributeError, KeyError, TypeError) as e:
                raise TypeError(f"Failed to manually instantiate filter from config {filter_config}: {e}")
            
        raise TypeError(f"Filter config must be either callable or a valid config dict with __target__, got {type(filter_config)}")

    def get_valid_intervals_from_filters(self, visualize: bool = False) -> List[TimeInterval]:
        valid_intervals = None
        for modality in self.modality_config:
            if "filters" in self.modality_config[modality]:
                device = self._experiment.devices[modality]
                for filter_name, filter_config in self.modality_config[modality]["filters"].items():
                    # Get the final callable filter function
                    filter_function = self._get_callable_filter(filter_config)
                    valid_intervals_ = filter_function(device_=device)
                    if visualize:
                        print(f"modality: {modality}, filter: {filter_name}")
                        visualization_string = get_stats_for_valid_interval(valid_intervals_, self.start_time, self.end_time)
                        print(visualization_string)
                    if valid_intervals is None:
                        valid_intervals = valid_intervals_
                    else:
                        valid_intervals = find_intersection_between_two_interval_arrays(valid_intervals, valid_intervals_)

        return valid_intervals
    
        
    def get_condition_mask_from_meta_conditions(self, valid_conditions_sum_of_product: List[dict]) -> np.ndarray:
        """Creates a boolean mask for trials that satisfy any of the given condition combinations.
        
        Args:
            valid_conditions_sum_of_product: List of dictionaries, where each dictionary represents a set of
                conditions that should be satisfied together (AND). Multiple dictionaries are combined with OR.
                Example: [{'tier': 'train', 'stim_type': 'natural'}, {'tier': 'blank'}] matches trials that
                are either (train AND natural) OR blank.

        Returns:
            np.ndarray: Boolean mask indicating which trials satisfy at least one set of conditions.
        """
        all_conditions = None
        for valid_conditions_product in valid_conditions_sum_of_product:
            conditions_of_product = None
            for k, valid_condition in valid_conditions_product.items():
                trial_conditions = self.meta_conditions[k]
                condition_mask = np.array([condition == valid_condition for condition in trial_conditions])
                if conditions_of_product is None:
                    conditions_of_product = condition_mask
                else:
                    conditions_of_product &= condition_mask
            if all_conditions is None:
                all_conditions = conditions_of_product
            else:
                all_conditions |= conditions_of_product
        return all_conditions
    
    def get_screen_sample_mask_from_meta_conditions(self, satisfy_for_next: int, valid_conditions_sum_of_product: List[dict], filter_for_valid_intervals: bool = True) -> np.ndarray:
        """Creates a boolean mask indicating which screen samples satisfy the given conditions.

        Args:
            satisfy_for_next: Number of consecutive samples that must satisfy conditions
            valid_conditions_sum_of_product: List of condition dictionaries combined with OR logic,
                where conditions within each dictionary use AND logic

        Returns:
            Boolean array matching screen sample times, True where conditions are met
        """
        all_conditions = self.get_condition_mask_from_meta_conditions(valid_conditions_sum_of_product)
        sample_mask = np.zeros_like(self._screen_sample_times, dtype=bool)
        valid_indices = np.where(all_conditions)[0]
        
        filter_valid_intervals = self.get_valid_intervals_from_filters(visualize=False) if filter_for_valid_intervals else None
        # filter_valid_intervals = None
        
        if len(valid_indices) > 0:
            starts = self._start_times[valid_indices]
            ends = self._end_times[valid_indices]
            
            # Create TimeIntervals from starts and ends
            trial_intervals = [TimeInterval(start, end) for start, end in zip(starts, ends)]
            
            # If we have filter_valid_intervals, find intersection with trial intervals
            if filter_valid_intervals:
                # Find intersection between trial intervals and filter valid intervals
                valid_intervals = find_intersection_between_two_interval_arrays(trial_intervals, filter_valid_intervals)
            else:
                valid_intervals = trial_intervals

            # Apply mask only for the intersected intervals
            for interval in valid_intervals:
                mask = (self._screen_sample_times >= interval.start) & (self._screen_sample_times < interval.end)
                sample_mask |= mask

        if satisfy_for_next > 1:
            windows = np.lib.stride_tricks.sliding_window_view(sample_mask, satisfy_for_next)
            sample_mask = np.all(windows, axis=1)

        return sample_mask

    def get_full_valid_sample_times(self, filter_for_valid_intervals: bool = True) -> Iterable:
        """
        iterates through all sample times and checks if they could be used as
        start times, eg if the next `self.chunk_sizes["screen"]` points are still valid
        based on the previous meta condition filtering
        :returns:
            valid_times: np.array of valid starting points
        """

        # Calculate all possible end indices
        chunk_size = self.chunk_sizes["screen"]
        n_samples = len(self._screen_sample_times) - chunk_size + 1
        possible_indices = np.arange(n_samples)
        
        # Check duration condition vectorized
        duration_mask = self._screen_sample_times[possible_indices + chunk_size - 1] < self.end_time

        # this assumes that the valid_condition is a single condition
        valid_conditions = [self.modality_config["screen"]["valid_condition"]]

        if self.modality_config["screen"]["include_blanks"]:
            additional_valid_conditions = {"tier": "blank"}  
            valid_conditions.append(additional_valid_conditions)

        sample_mask_from_meta_conditions = self.get_screen_sample_mask_from_meta_conditions(chunk_size, valid_conditions, filter_for_valid_intervals)

        final_mask = duration_mask & sample_mask_from_meta_conditions

        ### added to ensure that only full chunks for all modalites get retuned

        # Check other modalities to ensure the timepoints fall within their measurement periods
        for modality in self.device_names:
            if modality == "screen":
                continue  # Skip screen (already handled)
                
            # Get start and end times for this modality
            modality_start, modality_end = self._experiment.get_valid_range(modality)
            
            # Check if each timepoint falls within the modality's measurement period
            # We need to check both the start and end of the chunk
            chunk_starts = self._screen_sample_times[possible_indices]
            chunk_ends = self._screen_sample_times[possible_indices + chunk_size - 1]
            
            # A timepoint is valid if both the start and end of the chunk fall within the modality's period
            modality_mask = (chunk_starts >= modality_start) & (chunk_ends <= modality_end)
            
            # Update final mask
            final_mask = final_mask & modality_mask

        return self._screen_sample_times[possible_indices[final_mask]]


    def shuffle_valid_screen_times(self) -> None:
        """
        Shuffle valid screen times using the dataset's random number generator
        for reproducibility.
        """
        times = self._full_valid_sample_times
        if self.seed is not None:
            self._valid_screen_times = np.sort(
                self._rng.choice(times, size=len(times) // self.sample_stride, replace=False)
            )
        else:
            self._valid_screen_times = np.sort(
                np.random.choice(times, size=len(times) // self.sample_stride, replace=False)
            )

    def get_data_key_from_root_folder(cls, root_folder):
        """
        Extract a data key from the root folder path by checking for a meta.json file.

        Args:
            root_folder (str or Path): Path to the root folder containing dataset

        Returns:
            str: The extracted data key or folder name if meta.json doesn't exist or lacks data_key
        """
        # Convert Path object to string if necessary
        root_folder = str(root_folder)

        # Construct the path to meta.json
        meta_file_path = os.path.join(root_folder, "meta.json")

        # Initialize meta as an empty dict
        meta = {}

        # Check if the file exists before trying to open it
        if os.path.isfile(meta_file_path):
            try:
                with open(meta_file_path, "r") as file:
                    meta = json.load(file)

                # Get data_key from meta if it exists
                if "data_key" in meta:
                    return meta["data_key"]
                elif "scan_key" in meta:
                    key = meta["scan_key"]
                    data_key = f"{key['animal_id']}-{key['session']}-{key['scan_idx']}"
                    return data_key
                if "dynamic" in root_folder:
                    dataset_name = path.split("dynamic")[1].split("-Video")[0]
                    return dataset_name
                elif "_gaze" in path:
                    dataset_name = path.split("_gaze")[0].split("datasets/")[1]
                    return dataset_name
                else:
                    print(f"No 'data_key' found in {meta_file_path}, using folder name instead")
            except json.JSONDecodeError:
                print(f"Error: {meta_file_path} is not a valid JSON file")
            except Exception as e:
                print(f"Error loading {meta_file_path}: {str(e)}")
        else:
            print(f"No metadata file found at {meta_file_path}")
        return os.path.basename(root_folder)


    def __len__(self):
        return len(self._valid_screen_times)

    def __getitem__(self, idx) -> dict:
        out = {}
        timestamps = {}
        s = self._valid_screen_times[idx]
        for device_name in self.device_names:
            sampling_rate = self.sampling_rates[device_name]
            chunk_size = self.chunk_sizes[device_name]
            chunk_s = chunk_size / sampling_rate

            times = np.linspace(s, s + chunk_s, chunk_size, endpoint=False)
            times = times + self.modality_config[device_name].offset
            data, _ = self._experiment.interpolate(times, device=device_name)
            out[device_name] = self.transforms[device_name](data).squeeze(0) # remove dim0 for response/eye_tracker/treadmill
            # TODO: find better convention for image, video, color, gray channels. This makes the monkey data same as mouse.
            if device_name == "screen":
                if out[device_name].shape[-1] == 3:
                    out[device_name] = out[device_name].permute(0, 3, 1, 2).contiguous()
                if out[device_name].shape[0] == chunk_size:
                    out[device_name] = out[device_name].transpose(0, 1).contiguous()

            if device_name == 'responses':
                if self._experiment.devices["responses"].use_phase_shifts:
                    phase_shifts = self._experiment.devices["responses"]._phase_shifts
                    times = (times - times.min())[:, None] + phase_shifts[None, :]

            timestamps[device_name] = torch.from_numpy(times)
        out["timestamps"] = timestamps
        if self.add_behavior_as_channels:
            out = add_behavior_as_channels(out)

        # return only the keys that were explicitly requested
        # (necessary when pin_memory=True in dataloader)
        out = {k: out[k] for k in self.out_keys if k in out}
        return out

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataset's RNG."""
        return {
            'rng_state': self._rng.get_state() if self.seed is not None else None,
            'valid_screen_times': self._valid_screen_times.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataset's RNG state."""
        if state['rng_state'] is not None and self.seed is not None:
            self._rng.set_state(state['rng_state'])
        self._valid_screen_times = state['valid_screen_times']
