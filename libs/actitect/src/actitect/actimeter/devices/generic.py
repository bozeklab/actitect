import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ... import utils
from ..basedevice import BaseDevice

__all__ = ['Generic']

logger = logging.getLogger(__name__)


class Generic(BaseDevice):
    """Subclass of BaseDevice to load data from device-agnostic CSV files or a preloaded pandas.DataFrame.

    CSV/DataFrame input is normalized to the canonical columns ``time``, ``x``, ``y`` and ``z``. Headerless CSVs
    are supported when the first column is datetime-like and columns 1-3 are numeric. Additional headerless columns
    are treated as auxiliary and ignored after a lightweight structural validation.
    """

    _HEADERLESS_SAMPLE_ROWS = 100

    def __init__(
            self,
            filepath: Optional[Path],
            patient_id: str,
            *,
            raw_df: Optional[pd.DataFrame] = None,
            header: Optional[dict] = None,
    ):
        super().__init__(filepath=filepath, patient_id=patient_id, raw_df=raw_df, header=header)

    def __str__(self):
        return f"Generic(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, resolve_duplicates: bool = True, header_only: bool = False):
        if header_only:
            raise ValueError("'header_only' is not supported for Generic device.")

        try:
            filepath = self.processing_info['loading']['filepath']

            delimiter = utils.detect_csv_delimiter(filepath)
            logger.debug(f'(io: {self.meta["patient_id"]}) detected delimiter: \'{delimiter}\'.')
            header_present, mapping, usecols = self._infer_csv_structure(filepath, delimiter)

            if header_present:
                df = pd.read_csv(filepath, sep=delimiter)
            else:
                df = pd.read_csv(
                    filepath,
                    sep=delimiter,
                    header=None,
                    usecols=usecols,
                    names=['time', 'x', 'y', 'z'],
                )

            df.rename(columns=str.lower, inplace=True)
            if mapping:
                df.rename(columns=mapping, inplace=True)

            required = ['time', 'x', 'y', 'z']
            missing = [column for column in required if column not in df.columns]
            if missing:
                raise RuntimeError(
                    f'CSV is missing canonical column(s) after normalization: {missing}. '
                    f'Available columns: {list(df.columns)}.'
                )

            # Parse explicitly instead of relying on read_csv(parse_dates=...). This guarantees that malformed
            # timestamps fail here with a useful CSV-parsing error rather than later when BaseDevice inspects the index.
            try:
                df['time'] = pd.to_datetime(df['time'], errors='raise')
            except Exception as e:
                raise RuntimeError(f"Unable to parse canonical 'time' column as datetime: {e}") from e

            for axis in ('x', 'y', 'z'):
                try:
                    df[axis] = pd.to_numeric(df[axis], errors='raise')
                except Exception as e:
                    raise RuntimeError(f"Unable to parse canonical '{axis}' column as numeric: {e}") from e

            df.set_index('time', inplace=True)
            df = df[['x', 'y', 'z']]

            if not isinstance(df.index, pd.DatetimeIndex):
                raise RuntimeError(
                    f"Canonical 'time' column did not produce a DatetimeIndex (got {type(df.index).__name__})."
                )

            self.status_ok = 1
            return df, {'sample_rate': None}

        except Exception as e:
            logger.error("Exception occurred while parsing CSV: %s", e)
            self.status_ok = 0
            raise RuntimeError(f"CSV parsing failed: {e}") from e

    @classmethod
    def _infer_csv_structure(cls, filepath: Path, delimiter: str):
        """Infer whether a CSV is headered and map it to ``time,x,y,z``.

        Headerless input is accepted only after a lightweight sample confirms that column 0 is datetime-like and
        columns 1-3 are numeric. If more than four columns are present, columns 4+ are treated as auxiliary and are
        ignored by the Generic loader.

        Returns
        -------
        tuple
            ``(header_present, mapping, usecols)``. ``usecols`` is only used for headerless input.
        """
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            first = f.readline().strip()

        if not first:
            raise RuntimeError('CSV is empty or its first line is empty.')

        tokens = [t.strip().lower() for t in first.split(delimiter)]

        # Canonical header.
        if tokens[:4] == ['time', 'x', 'y', 'z']:
            return True, None, None

        # A row that starts with datetime-like + three numeric values is a candidate headerless CSV. Do not infer
        # semantics from the first row alone: validate a small sample before accepting the fallback.
        if cls._looks_like_headerless_data_row(tokens):
            return cls._infer_headerless_structure(filepath, delimiter)

        logger.warning(f'CSV header not canonical ({tokens}). Attempting to map to canonical columns.')

        mapping = {}

        time_aliases = {'time', 'timestamp', 'datetime', 'date'}
        x_aliases = {'x', 'acc_x', 'ax', 'accel_x', 'a_x'}
        y_aliases = {'y', 'acc_y', 'ay', 'accel_y', 'a_y'}
        z_aliases = {'z', 'acc_z', 'az', 'accel_z', 'a_z'}

        for token in tokens:
            if token in time_aliases:
                mapping[token] = 'time'
            elif token in x_aliases:
                mapping[token] = 'x'
            elif token in y_aliases:
                mapping[token] = 'y'
            elif token in z_aliases:
                mapping[token] = 'z'

        required = {'time', 'x', 'y', 'z'}

        if set(mapping.values()) != required:
            raise RuntimeError(f'Unable to map CSV header {tokens} to canonical columns: time,x,y,z.')

        return True, mapping, None

    @staticmethod
    def _looks_like_headerless_data_row(tokens):
        """Cheap first-row check before reading a validation sample."""
        if len(tokens) < 4:
            return False

        try:
            parsed_time = pd.to_datetime(tokens[0], errors='coerce')
            if pd.isna(parsed_time):
                return False
            float(tokens[1])
            float(tokens[2])
            float(tokens[3])
            return True
        except (TypeError, ValueError):
            return False

    @classmethod
    def _infer_headerless_structure(cls, filepath: Path, delimiter: str):
        """Validate and describe a candidate headerless CSV without loading the full file."""
        sample = pd.read_csv(
            filepath,
            sep=delimiter,
            header=None,
            nrows=cls._HEADERLESS_SAMPLE_ROWS,
            dtype=str,
        )

        n_columns = sample.shape[1]
        if n_columns < 4:
            raise RuntimeError(
                f'Headerless CSV has only {n_columns} columns; at least 4 are required for time,x,y,z.'
            )

        time_values = sample.iloc[:, 0]
        parsed_time = pd.to_datetime(time_values, errors='coerce')
        invalid_time = parsed_time.isna()
        if invalid_time.any():
            examples = time_values.loc[invalid_time].head(5).tolist()
            raise RuntimeError(
                f'Headerless CSV inference failed: column 0 is not consistently datetime-like in the first '
                f'{len(sample)} rows ({int(invalid_time.sum())} invalid value(s); examples: {examples}).'
            )

        for column_index, axis in zip((1, 2, 3), ('x', 'y', 'z')):
            values = sample.iloc[:, column_index]
            parsed = pd.to_numeric(values, errors='coerce')
            invalid = parsed.isna()
            if invalid.any():
                examples = values.loc[invalid].head(5).tolist()
                raise RuntimeError(
                    f"Headerless CSV inference failed: column {column_index}, inferred as '{axis}', is not "
                    f'consistently numeric in the first {len(sample)} rows '
                    f'({int(invalid.sum())} invalid value(s); examples: {examples}).'
                )

        if n_columns == 4:
            logger.warning(
                'CSV has no header; inferred columns 0-3 as canonical time,x,y,z from sampled values.'
            )
        else:
            logger.warning(
                'CSV has no header and contains %d columns; inferred columns 0-3 as time,x,y,z from sampled '
                'values and ignoring %d auxiliary column(s).',
                n_columns,
                n_columns - 4,
            )

        # Limiting usecols is important: passing four names for a wider CSV can make pandas reinterpret leading
        # fields as an index. Reading only the canonical columns avoids that ambiguity entirely.
        return False, None, [0, 1, 2, 3]
