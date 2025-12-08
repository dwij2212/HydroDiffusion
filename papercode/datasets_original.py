"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from pathlib import PosixPath
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .datautils import (load_attributes, load_discharge, load_forcing, normalize_features, normalize_features_noprecip, 
                        reshape_data)


class CamelsTXT(Dataset):
    """PyTorch dataset to work with the raw text files in the CAMELS data set.
       
    Supports sliding windows of length seq_length followed by forecast_horizon days.
    Applies normalization differently depending on model type: full normalization for LSTM & diffusion, omit precip for MC-LSTM/MCR-LSTM.
    """

    def __init__(
        self,
        camels_root: PosixPath,
        basin: str,
        dates: List[pd.Timestamp],
        is_train: bool,
        seq_length: int = 365,
        forecast_horizon: int = 7,
        model_name: str = 'lstm',
        with_attributes: bool = False,
        attribute_means: pd.Series = None,
        attribute_stds: pd.Series = None,
        concat_static: bool = False,
        db_path: str = None
    ):
        self.camels_root = camels_root
        self.basin = basin
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.model_name = model_name.lower()
        self.is_train = is_train
        self.dates = dates
        self.with_attributes = with_attributes
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.concat_static = concat_static
        self.db_path = db_path

        self.q_std = None
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        # Load and preprocess
        self.x, self.y = self._load_data()
        if self.with_attributes:
            self.attributes = self._load_attributes()
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.with_attributes:
            if self.concat_static:
                repeated = self.attributes.repeat((self.seq_length, 1))
                x = torch.cat([self.x[idx], repeated], dim=-1)
                return x, self.y[idx]
            else:
                return self.x[idx], self.attributes, self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load forcing and discharge
        df, area = load_forcing(self.camels_root, self.basin)
        df['QObs(mm/d)'] = load_discharge(self.camels_root, self.basin, area)

        # Slice for seq + horizon
        start = self.dates[0]
        end = self.dates[1] + pd.Timedelta(days=self.forecast_horizon)
        df = df[start:end]
        self.period_start, self.period_end = df.index[0], df.index[-1]

        # Build raw input matrix
        prcp = df['PRCP(mm/day)'].values.reshape(-1, 1)
        other = np.stack([
            df['SRAD(W/m2)'].values,
            df['Tmax(C)'].values,
            df['Tmin(C)'].values,
            df['Vp(Pa)'].values
        ], axis=1)

        y = df['QObs(mm/d)'].values.reshape(-1, 1)
        
        # Normalize x
        if self.model_name in ('lstm', 'diffusion_lstm'):
            X = np.concatenate([prcp, other], axis=1)
            X = normalize_features(X, variable='inputs')
        elif self.model_name in ('mclstm', 'mcrlstm'):
            other_norm = normalize_features_noprecip(other, variable='inputs')
            X = np.concatenate([prcp, other_norm], axis=1)
        else:
            raise ValueError(f"Unknown model_name '{self.model_name}'")

        # Reshape
        x, y = reshape_data(
            X, y,
            seq_length=self.seq_length,
            horizon=self.forecast_horizon
        )

        if self.is_train:
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(
                    f"Deleted {np.sum(np.isnan(y))} of {len(y)} records because of NaNs in basin {self.basin}"
                )
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
                
        q_std = np.std(y)
        # Normalize y
        y = normalize_features(y, variable='output')
        
        # To torch
        X_t = torch.from_numpy(x.astype(np.float32))
        Y_t = torch.from_numpy(y.astype(np.float32))
        return X_t, Y_t

    def _load_attributes(self) -> torch.Tensor:
        df = load_attributes(self.db_path, [self.basin], drop_lat_lon=True)
        df = (df - self.attribute_means) / self.attribute_stds
        self.attribute_names = df.columns
        attrs = df.loc[self.basin].values.astype(np.float32)
        return torch.from_numpy(attrs)


class CamelsH5(Dataset):
    def __init__(
        self,
        h5_file: PosixPath,
        basins: List[str],
        db_path: str,
        concat_static: bool = False,
        cache: bool = False,
        no_static: bool = False,
        model_name: str = 'lstm',
        forecast_horizon: int = 7,
        include_dates: bool = False
    ):
        self.h5_file = h5_file
        self.basins = basins
        self.db_path = db_path
        self.concat_static = concat_static
        self.cache = cache
        self.no_static = no_static
        self.model_name = model_name.lower()
        self.forecast_horizon = forecast_horizon
        self.include_dates = include_dates

        # Load and normalize static attributes
        self._load_attributes()

        if self.cache:
            self.x, self.y, self.sample_2_basin, self.q_stds, self.dates = self._preload_data()
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(self.h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            bas = [b.decode("ascii") for b in f["sample_2_basin"][:]]
            q = f["q_stds"][:]
            if "dates" in f:
                dates = [d.decode("ascii") for d in f["dates"][:]]
            else:
                dates = None
        return x, y, bas, q, dates

    def _load_attributes(self):
        df = load_attributes(self.db_path, self.basins, drop_lat_lon=True)
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()
        df_norm = (df - self.attribute_means) / self.attribute_stds
        self.attribute_names = df_norm.columns
        self.df = df_norm

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][idx]
            y = f["target_data"][idx]
            basin = f["sample_2_basin"][idx].decode("ascii")
            q_std = f["q_stds"][idx]
            date = f["dates"][idx].decode("ascii") if self.include_dates and "dates" in f else ""

        attrs = None
        if not self.no_static:
            arr = self.df.loc[basin].values.astype(np.float32)
            attrs = torch.from_numpy(arr)

        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        q_t = torch.tensor(q_std, dtype=torch.float32)

        if self.no_static:
            return x_t, y_t, q_t, basin, date
        else:
            return x_t, attrs, y_t, q_t, basin, date

    def get_attribute_means(self) -> pd.Series:
        return self.attribute_means

    def get_attribute_stds(self) -> pd.Series:
        return self.attribute_stds

    def get_static_attributes_tensor(self, basin_ids: List[str]) -> torch.Tensor:
        rows = np.array([self.df.loc[b].values for b in basin_ids])
        return torch.from_numpy(rows.astype(np.float32))