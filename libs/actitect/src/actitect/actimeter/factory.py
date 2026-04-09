import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .. import utils
from .devices import ActiGraph, Axivity, Generic, GENEActiv

__all__ = ['ActimeterFactory', 'SUPPORTED_FILETYPES']

logger = logging.getLogger(__name__)

EXT_2_DEVICE_MAP = {'.cwa': Axivity, '.bin': GENEActiv, '.gt3x': ActiGraph, '.csv': Generic}
DEVICE_KWARGS_MAP = {Axivity: {'legacy_mode'}, GENEActiv: set(), ActiGraph: set(), Generic: set()}
SUPPORTED_FILETYPES = sorted(list(EXT_2_DEVICE_MAP.keys()))


class ActimeterFactory:
    """ Entry point to all actimeter devices to automatically initialize the specific device based on file extension.
    Currently supported devices are:
        - Axivity Ax6: .cwa files
        - GENEActiv: .bin files
        - Actigraph: .gt3x files
        - Generic: .csv files or preloaded DataFrames"""

    def __new__(cls, data: Union[str, Path, pd.DataFrame], patient_id: str, mute: bool = False, **kwargs):
        """ Return an instance of the appropriate actimeter device subclass based on the file extension.
        Parameters:
            :param data: (Union[str, Path, DataFrame]) The path to the data file for the actimeter device, if a
                DataFrame is provided, the preloaded data will be used, and we fall back to Generic device.
            :param patient_id: (str) The patient identifier associated with the actimeter data.
            :param mute: (bool, Optional) If True, mutes the logging. Defaults to False.
        Returns:
            :return: (BaseDevice) instance of base device su corresponding to given actimeter type."""

        if isinstance(data, pd.DataFrame):
            device_class = Generic
            valid_keys = DEVICE_KWARGS_MAP.get(device_class, set())
            header = kwargs.pop('header', None)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            if not mute:
                logger.info(f"(io: {patient_id}) detected '{device_class.__name__}' device (preloaded DataFrame).")
                ignored = set(kwargs) - valid_keys
                if ignored:
                    logger.warning(f"(io: {patient_id}) Ignored unsupported kwargs for "
                                   f"{device_class.__name__}: {ignored}")

            return device_class(
                filepath=None, patient_id=patient_id, raw_df=data.copy(), header=header, **filtered_kwargs)

        assert Path(data).is_file(), f"received file path ('{data}') is not a file."
        _ext = utils.get_file_extension(data)
        device_class = EXT_2_DEVICE_MAP.get(_ext)

        if device_class is None:
            raise ValueError(
                f"file extension '{_ext}' is not supported. "
                f"Available devices are: { {k: v.__name__ for k, v in EXT_2_DEVICE_MAP.items()} }."
            )
        else:

            valid_keys = DEVICE_KWARGS_MAP.get(device_class, set())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

            if not mute:
                logger.info(f"(io: {patient_id}) detected '{device_class.__name__}' device.")
                ignored = set(kwargs) - valid_keys
                if ignored:
                    logger.warning(f"(io: {patient_id}) Ignored unsupported kwargs for"
                                   f"{device_class.__name__}: {ignored}")

            return device_class(data, patient_id, **filtered_kwargs)
