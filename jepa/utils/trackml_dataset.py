import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
from pathlib import Path
from typing import Union, Optional, Callable

class TrackMLDataset(Dataset):
    r"""
    A Dataset subclass for the TrackML dataset in HDF5 format.

    Args:
        file (str or Path): path to the HDF5 file holding the data.
        scaling_factor (float, optional): a multiplicative scaling factor applied to the hit positions (default: ``1.0``).
            Note that, by default, positions are specified in millimeters.
        transform (Callable, optional): a function used to further process the output.
        float_dtype (torch.dtype, optional): the dtype of the returned tensors for floating-point features (default: ``torch.float32``).
    """

    def __init__(
        self,
        file: Union[str,Path],
        scaling_factor: float=1.0,
        transform: Optional[Callable]=None,
        float_dtype=torch.float32,
    ):
        super(TrackMLDataset).__init__()
        self.file = h5py.File(file, 'r')
        self.number_of_events = self.file.attrs['number_of_events']
        self.hits = self.file['hits']
        self.truth = self.file['truth']
        self.float_dtype = float_dtype
        self.scaling_factor = torch.tensor(scaling_factor, dtype=float_dtype)
        self.transform = transform

    def __del__(self):
        self.file.close()

    def __len__(self):
        return self.number_of_events

    def __getitem__(self, idx: int):
        x, hit_id = self._get_hits(idx)
        pids = self._get_particle_ids(idx, hit_id)
        output = {
            'x': x,
            'mask': torch.ones(x.shape[0], dtype=bool),
            'pids': pids,
            'event': None,
        }
        if self.transform:
            output = self.transform(output)
        return output

    def _get_hits(self, idx: int):
        offset = self.hits['event_offset'][idx]
        length = self.hits['event_length'][idx]
        event_slice = slice(offset, offset+length)
        hit_id = pd.DataFrame({'hit_id': self.hits['hit_id'][event_slice]}, copy=False).set_index('hit_id')
        x = torch.zeros((length, 3), dtype=self.float_dtype)
        x[:,0] = torch.from_numpy(self.hits['x'][event_slice]) * self.scaling_factor
        x[:,1] = torch.from_numpy(self.hits['y'][event_slice]) * self.scaling_factor
        x[:,2] = torch.from_numpy(self.hits['z'][event_slice]) * self.scaling_factor
        return x, hit_id

    def _get_particle_ids(self, idx: int, detected_hits: pd.DataFrame):
        # Note: not all hits in "hits" are also in "truth", and reciprocally
        # Note: the weight is ignored for now
        offset = self.truth['event_offset'][idx]
        length = self.truth['event_length'][idx]
        event_slice = slice(offset, offset+length)
        truth = pd.DataFrame({
            'hit_id': self.truth['hit_id'][event_slice],
            'particle_id': self.truth['particle_id'][event_slice],
        }, copy=False).set_index('hit_id')
        # Let’s find the true particle_id corresponding to each detected hit_id
        joined = detected_hits.join(truth, on='hit_id', how='inner')
        assert joined['particle_id'].dtype == truth['particle_id'].dtype
        matched_particle_id = torch.from_numpy(joined['particle_id'].values)
        return matched_particle_id