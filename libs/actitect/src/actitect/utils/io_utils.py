import ast
import csv
import datetime
import json
import logging.config
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Optional, Union, Any, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    'get_experiment_root', 'load_yaml_file', 'dump_yaml_file', 'detect_csv_delimiter', 'read_meta_csv_to_df',
    'dump_to_json', 'read_from_json', 'get_file_extension', 'check_make_dir', 'str_to_snake_case',
    'resolve_root_or_abs_path'
]


def check_make_dir(dir_path: Path, use_existing=False, verbose=True, time_extension: bool = True):
    """
    Ensure the directory at `dir_path` exists. If not, create it. If it exists and `use_existing` is False,
    a new directory with an incremented suffix ('_1', '_2', etc.) is created.

    Parameters:
        :param: dir_path: (Path) path object of the directory to check/create.
        :param: use_existing: (bool, Optional) If True, use the existing directory if it exists.
                    If False, create a new directory with a numeric suffix if it exists. Default is False.
        :param: verbose: (bool, Optional) If True, logs messages about the directory creation or reuse. Default is True.
        :param: time_extension (bool, Optional) If True, will add current time as extension instead of 1,2,3...

    Returns:
        :return: (Path) the path objected pointing to the created or reused directory.

    Notes:
    - If the directory already exists and `use_existing` is True, the existing directory is reused.
    - If the directory already exists and `use_existing` is False, a new directory with a suffix is created.
    """
    if not dir_path.exists():
        try:
            # os.makedirs(dir_path)
            dir_path.mkdir(parents=True, exist_ok=False)
            if verbose:
                logger.debug(f"(io): dir. '{dir_path}' created.")
            return dir_path
        except OSError as e:
            logger.error(f"(io): error creating dir. '{dir_path}': {e}")
    else:
        if use_existing:
            if verbose:
                logger.info(f"(io): dir '{dir_path}' exists and will be used."
                            f"Set 'use_existing' to False, to create new dir to avoid potential overwriting.")
            return dir_path

        else:

            if time_extension:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
                new_dir_name = f"{dir_path.name}_{timestamp}"
                new_dir_path = dir_path.parent / new_dir_name
                try:
                    new_dir_path.mkdir(parents=True, exist_ok=False)
                    if verbose:
                        logger.warning(f"(io): Directory '{dir_path}' already exists. Created '{new_dir_path}'.")
                    return new_dir_path
                except OSError as e:
                    logger.error(f"(io): Error creating directory '{new_dir_path}': {e}")
                    raise

            # Determine the numeric suffix to append to the directory name
            suffix = 1
            # new_dir_path = f"{dir_path}_{suffix}"
            new_dir_path = dir_path.parent.joinpath(f"{dir_path.name}_{suffix}")

            # while os.path.exists(new_dir_path):
            while new_dir_path.exists():
                suffix += 1
                # new_dir_path = f"{dir_path}_{suffix}"
                new_dir_path = dir_path.parent.joinpath(f"{dir_path.name}_{suffix}")
            try:
                # os.makedirs(new_dir_path)
                new_dir_path.mkdir(parents=True, exist_ok=False)
                logger.warning(f"(io): dir. '{dir_path}' already exists. created {new_dir_path}")
                return new_dir_path
            except OSError as e:
                logger.error(f"(io): error creating dir. '{new_dir_path}': {e}")
                raise


def get_experiment_root() -> Path:
    from importlib.resources import files
    cfg_dir = files('actitect.config')
    for name in ('experiment_root.local.json', 'experiment_root.json'):
        p = cfg_dir / name
        try:
            with p.open("r", encoding="utf-8") as f:
                root = (json.load(f).get("EXPERIMENT_ROOT") or "").strip()
            if root:
                return Path(root)
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "Experiment root is unset. Create 'experiment_root.local.json' via install.py "
        "or fill 'experiment_root.json' with a valid path.")


def load_yaml_file(path):
    with open(path, "r") as stream:
        try:
            _cfg = yaml.safe_load(stream)
            return _cfg
        except yaml.YAMLError as exc:
            logger.error(f"(io): error ({exc}) encountered while loading config {path}")


def dump_yaml_file(data: Any, file_path: Union[str, Path], *, sort_keys: bool = False):
    file_path = Path(file_path)

    try:
        with open(file_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=sort_keys, default_flow_style=False)
    except yaml.YAMLError as exc:
        logger.error(f"(io): error ({exc}) encountered while writing yaml file {file_path}")
        raise


def detect_csv_delimiter(filepath: Union[str, Path], *, sample_size: int = 1 << 14,
                         possible: Optional[List[str]] = None, fallback: str = ';'):
    """Detect the delimiter of a csv file.
    Parameters
        :param filepath: (str) | pathlib.Path
        :param sample_size: (int) bytes of the file to inspect (default 16 kB)
        :param possible: (str | list) set of delimiters to test, e.g. ',;\t|', default = ',;\t|:'
        :param fallback: (str) delimiter returned when nothing can be decided.
    Returns
        :return: (str) detected delimiter or *fallback* as last resort."""
    path = Path(filepath)

    if isinstance(possible, str):
        candidates = list(possible)
    elif isinstance(possible, (list, tuple, set)):
        candidates = list(possible)
    else:
        candidates = [',', ';', '\t', '|', ':']

    # 1. collect test sample
    with path.open('rb') as f:
        raw = f.read(sample_size)
    sample = raw.decode("utf-8", errors="ignore")
    if not sample:
        logger.warning("empty file – falling back to default delimiter %r", fallback)
        return fallback

    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if len(lines) < 2:  # at least two real lines are needed
        logger.warning("not enough lines for delimiter detection – defaulting")
        return fallback

    # 2. try csv.Sniffer but verify proposed delimiter
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=''.join(candidates))
        delim = sniffed.delimiter
        if delim in candidates:
            # verify: every line must have identical, non-zero column count
            counts = [ln.count(delim) for ln in lines]
            if counts and counts.count(counts[0]) == len(counts) and counts[0] > 0:
                return delim
            logger.debug("Sniffer proposed %r but column counts inconsistent – discarding", delim)
    except Exception as err:
        logger.debug("csv.Sniffer failed (%s) – trying heuristics", err)

    # 3. heuristic scoring: score = (low variance of counts) + (high mean count)
    best_delim, best_score = None, -math.inf
    for delim in candidates:
        counts = [ln.count(delim) for ln in lines]
        if not counts or all(c == 0 for c in counts):
            continue  # delimiter never appears
        # variance should be low, mean should be high
        try:
            var = statistics.pvariance(counts)
        except statistics.StatisticsError:  # happens when len(counts)==1
            var = 0.0
        score = (sum(counts) / len(counts)) - var  # larger → better
        if score > best_score:
            best_delim, best_score = delim, score
    if best_delim:
        logger.debug("Heuristic chose delimiter %r (score %.3f)", best_delim, best_score)
        return best_delim

    # 4. failed, fallback
    logger.warning("failed to detect delimiter – defaulting to %r", fallback)
    return fallback


def read_meta_csv_to_df(path_to_csv: Path, exclude: bool = False, verbose: bool = True):
    def _row_has_valid_filename(row):
        from ..actimeter import SUPPORTED_FILETYPES
        return any(str(row['filename']).endswith(ext) for ext in SUPPORTED_FILETYPES)

    def _handle_exclusion(df: pd.DataFrame, *, col: str = 'exclude',
                          drop: bool = True,
                          log: bool = True,
                          ) -> pd.DataFrame:
        """
        Canonicalise *df[col]* and (optionally) drop rows that have exclude == '1'.

        Canonical forms
        ---------------
            keep   : 0, 0.0, '0', '0.0', 'false', 'no',  NaN/None/''  → '0'
            maybe  : '?', ' ?', ' ? '                                 → '?'
            drop   : 1, 1.0, '1', '1.0', 'true', 'yes'                → '1'
        """

        # ─────────────────────── helper ────────────────────────────────
        def _canonical(val: Any) -> object:
            if pd.isna(val):  # NaN / None / ''
                return '0'  # treat as “keep”
            if isinstance(val, (int, np.integer, float, np.floating)):
                return '1' if float(val) == 1.0 else '0'
            s = str(val).strip().lower()
            if s in {"1", "1.0", "true", "yes"}:
                return '1'
            if s in {"0", "0.0", "false", "no"}:
                return '0'
            if s == '?':
                return '?'
            return '0'  # everything else ⇒ keep

        # ─────────────────── canonicalise column ───────────────────────
        out = df.copy()
        if col not in out.columns:
            out[col] = '0'  # default: keep
        out[col] = out[col].apply(_canonical).astype('string')

        # ──────────────────── log always, drop optional ────────────────
        flagged = out[out[col] == '1']
        if log and len(flagged):
            ids = flagged['ID'].tolist() if 'ID' in flagged.columns else flagged.index.tolist()
            status = "will be dropped" if drop else "will be kept"
            logger.info(f"Found {len(flagged)} row(s) with exclude == 1 ({status}). IDs: {ids}")

        if drop:
            out = out[out[col] != '1']

        return out

    def _assign_record_ID(group: pd.DataFrame, ID_value: str):
        """ Assigns unique recording IDs for patients with multiple recordings.
        Preserves user-defined values where present. """
        group.insert(group.columns.get_loc('filename') + 1, 'ID', ID_value)
        if group.shape[0] == 1:
            return group  # only one recording for ID, no change needed
        if group['record_ID'].isna().all():  # multiple recordings but no user-define record_IDs
            group['record_ID'] = [f"rec{i + 1}" for i in range(group.shape[0])]  # auto-complete record_IDs
        elif group['record_ID'].isna().any():  # some exist, but not all
            logger.warning(f"`record_ID` values are only partially defined for ID={ID_value}."
                           f"Using auto-completion for missing values. ")
            existing_ids = set(group['record_ID'].dropna())  # keep defined IDs
            new_ids = [f"rec{i + 1}" for i in range(1, group.shape[0] + 1) if f"rec{i + 1}" not in existing_ids]
            group.loc[group['record_ID'].isna(), 'record_ID'] = new_ids[:group['record_ID'].isna().sum()]

        return group

    def _summary_note_mask(
            df: pd.DataFrame,
            *,
            min_nonempty: int = 2,  # keep rows with at least this many non-empty cells
            numeric_match_frac: float = 0.8,  # row matches ≥ this frac of numeric column aggregates
            tokens: Optional[Iterable[str]] = None,  # tokens marking totals/notes, any language
            tol: float = 1e-9,
    ) -> pd.Series:
        """
        Heuristics to flag Excel-style totals/subtotals/notes rows without assuming ID patterns.
        Flags a row if ANY of:
          (A) 'totals/notes' tokens present,
          (B) row matches column aggregates (sum/count) in ≥ numeric_match_frac numeric cols,
          (C) row is mostly empty (non-empty cells < min_nonempty).
        """
        if tokens is None:
            tokens = ("total", "subtotal", "sum", "gesamt", "Σ", "note", "notes", "comment")

        token_patterns = [
            re.compile(rf"(?<!\w){re.escape(str(token))}(?!\w)", flags=re.IGNORECASE)
            for token in tokens
        ]

        str_df = df.astype(object).where(
            ~df.apply(lambda col: col.map(lambda x: isinstance(x, (list, dict, set))))
        )

        token_hit = str_df.apply(
            lambda row: any(
                isinstance(value, str) and any(pattern.search(value) for pattern in token_patterns)
                for value in row
            ),
            axis=1,
        )

        # (B) numeric-aggregate match (sum or count)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            sums = df[num_cols].sum(numeric_only=True)
            counts = df[num_cols].notna().sum()

            def _match_frac(row: pd.Series) -> float:
                eq_sum = (row[num_cols] - sums).abs() <= tol
                eq_cnt = (row[num_cols] - counts).abs() <= tol
                return np.nanmean((eq_sum | eq_cnt).to_numpy())  # fraction matched

            frac = df.apply(_match_frac, axis=1)
            agg_match = frac.fillna(0) >= numeric_match_frac
        else:
            agg_match = pd.Series(False, index=df.index)

        # (C) mostly empty row
        nonempty_per_row = df.apply(lambda r: sum(str(v).strip() != "" and pd.notna(v) for v in r), axis=1)
        mostly_empty = nonempty_per_row < min_nonempty

        return token_hit | agg_match | mostly_empty

    delimiter = detect_csv_delimiter(path_to_csv)
    meta_df = pd.read_csv(path_to_csv, delimiter=delimiter)

    meta_df.columns = meta_df.columns.str.strip()
    if '#' in meta_df.columns:
        meta_df.set_index('#', inplace=True)

    # format timestamps if needed
    if 'timestamps' in meta_df.columns:
        meta_df['timestamps'] = meta_df['timestamps'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    filtered_cols = [_col for _col in meta_df.columns if 'Unnamed' not in _col]
    meta_df = meta_df[filtered_cols]

    # strip whitespaces in strings if present
    meta_df = meta_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # optionally exclude rows
    meta_df = _handle_exclusion(meta_df, drop=exclude, log=verbose)

    # ensure required columns exist:
    required_cols = {'filename', 'ID', 'diagnosis'}
    missing_cols = required_cols - set(meta_df.columns)
    if missing_cols:
        raise ValueError(f"Metadata file is missing required columns: {missing_cols}")
    if 'record_ID' not in meta_df.columns:
        meta_df['record_ID'] = None

    mask = _summary_note_mask(meta_df, min_nonempty=2, numeric_match_frac=.8)
    if mask.any():
        logger.warning(f"Detected {mask.sum()} potential summary row(s) in metadata CSV;"
                       f" they will be dropped – verify if intended.")
        meta_df = meta_df.loc[~mask].copy()

    meta_df = meta_df.groupby('ID', group_keys=False).apply(
        lambda g: _assign_record_ID(g, g.name), include_groups=False)

    # filename normalization and duplicate check
    meta_df['filename'] = meta_df['filename'].astype('string').str.strip().replace(
        {'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA})

    dupe_mask = meta_df['filename'].notna() & meta_df['filename'].duplicated(keep=False)
    if dupe_mask.any():
        dup_list = meta_df.loc[dupe_mask, 'filename'].unique()
        raise AssertionError(
            f"Duplicate filenames found in meta file (missing values ignored): {', '.join(map(str, dup_list))}")

    assert not meta_df.duplicated(subset=['ID', 'record_ID']).any(), \
        "Duplicate combinations of 'ID' and 'record_ID' found in meta file."
    return meta_df


class CustomJsonEncoder(json.JSONEncoder):
    """Serializes numpy and pandas objects as JSON."""
    try:
        from skopt.space import Dimension as _SkoptDimension
    except Exception:
        _SkoptDimension = None

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, pd.Timestamp):
            if pd.isna(obj):  # Handle NaT (missing datetime)
                return None
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None  # Serialize NaN as JSON null
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif self._SkoptDimension is not None and isinstance(obj, self._SkoptDimension):
            return self._serialize_dimension(obj)
        elif callable(obj):
            return str(obj)
        else:
            return super().default(obj)

    @staticmethod
    def _serialize_dimension(obj):
        """Serialize skopt Dimension objects to a dictionary."""
        serialized = {'name': obj.name, 'kwargs': {'type': obj.__class__.__name__}}
        for attribute in dir(obj):
            # filter out private and built-in attributes
            if not attribute.startswith('_') and not callable(getattr(obj, attribute)):
                serialized['kwargs'][attribute] = str(getattr(obj, attribute))
        return serialized


def dump_to_json(data_dict, file_path, indent=4, json_encoder=CustomJsonEncoder):
    """
    Save a dictionary to a JSON file.

    Parameters::
        :param data_dict: (dict) Dictionary to be saved.
        :param file_path: (str or Path) Path to the file where the JSON will be saved.
        :param indent: (int, optional) Indentation level for the JSON file. Default is 4.
        :param json_encoder: (json.JSONEncoder, optional) Custom JSON encoder, if needed.
    """
    file_path = Path(file_path)
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=indent, cls=json_encoder)


def read_from_json(file_path):
    """
    Read a saved JSON file as dict.

    Parameters::
        :param file_path: (str or Path) Path to the file where the JSON will be saved.
    """
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        _dict = json.load(f)
    return _dict


def get_file_extension(file_path, *, include_dot=True, all_extensions=False):
    """ For a given file path, return the file extension(s).

    Parameters:
        :param file_path: (Path or str): The path of the file.
        :param include_dot: (bool, Optional) If True, include the leading dot in the return str. Default is True.
        :param all_extensions: (bool, Optional) If True, returns all extensions as list.
            Default is False: return only last extension as str.
    Returns:
        :return: str or list[str] containing the last or all file extensions.
    """
    # Ensure file_path is a Path object
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if all_extensions:  # collect all suffixes
        suffixes = file_path.suffixes
        if not include_dot:
            suffixes = [suffix.lstrip('.') for suffix in suffixes]
        return suffixes
    else:  # only last suffix
        suffix = file_path.suffix
        if not include_dot:
            suffix = suffix.lstrip('.')
        return suffix


def list_subdirectories(base_path: Path, depth: int, stop_names: set = None):
    """  List subdirectories of base_path at a specified depth or earlier if a directory's name
    is in stop_names. When a directory is encountered whose name is in stop_names, that directory
    is added to the result and its children are not searched.
    Parameters:
      base_path (Path): The starting directory.
      depth (int): The maximum depth to search (depth=1 returns immediate children).
      stop_names (set, optional): A set of directory names at which to stop recursion.
                                  For example: {"early_stopping", "no_early_stopping"}
    Returns:
      list of Path: Subdirectories found, expressed as paths relative to base_path. """
    stop_names = stop_names or set()
    result = []

    def _traverse(path: Path, current_depth: int):
        # If we've reached the maximum depth or the directory name is in stop_names,
        # add it to the result (if it's a directory) and do not descend further.
        if current_depth == depth or path.name in stop_names:
            if path.is_dir():
                result.append(path)
            return
        # Otherwise, traverse its children.
        for child in path.iterdir():
            if child.is_dir():
                _traverse(child, current_depth + 1)

    _traverse(base_path, 0)
    # Convert the absolute paths in result to paths relative to base_path.
    return [p.relative_to(base_path) for p in result]


def str_to_snake_case(text: str):
    """" transform a given str to snake_case format """
    return re.sub(r'\s+', '_', text.strip().lower())


def resolve_root_or_abs_path(root_dir: str, maybe_path: str) -> Path:
    """Backwards-compatible path resolver:
    - absolute paths are used as-is
    - relative paths are interpreted relative to root_dir."""
    p = Path(maybe_path)
    return p if p.is_absolute() else Path(root_dir).joinpath(p)
