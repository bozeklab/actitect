
<p align="center">
  <img src="docs/logo.svg" alt="ActiTect logo" height="140" style="vertical-align:middle; margin-right:10px;">
</p>

# ActiTect & RBDisco: Actigraphy Toolkit and RBD Prediction

---

This repository hosts ActiTect, a general-purpose Python toolkit for actigraphy analysis(preprocessing,
feature extraction, QC, and visualization), and RBDisco, an extension that provides reproducible pipelines and 
pretrained models for RBD prediction from wrist actigraphy as described [here](https://arxiv.org/abs/2511.05221).
<!-- [Introduction](#introduction)-->
- [Overview](#overview)
- [Installation](#installation)
- [ActiTect Usage](#actitect-usage-) ([CLI](#actitect-cli-usage-) / [API](#actitect-api-usage))
- [RBDisco Usage](#rbdisco-usage) ([CLI](#rbdisco-usage) / [API](#rbdisco-api-usage))
- [Citation](#citation)

---

## Overview 
ActiTect is a modular, open-source toolkit for standardized analysis of research-grade actigraphy data.
It provides a unified command-line and Python interface for loading and preprocessing data from commonly 
used wearable devices. Core processing steps include resampling, bandpass filtering, auto-calibration,
artifact removal, and sleep/non-wear segmentation, enabling harmonized analyses across cohorts and studies.
The RBDisco plugin extends ActiTect with pretrained machine-learning models for the detection of 
REM Sleep Behavior Disorder (RBD) from actigraphy recordings.

<figure>
  <img src="docs/overview.png" alt="Pipeline overview" style="width:500px;">
  <figcaption><em>Figure 1 – ActiTect/RBDisco overview: full description at https://arxiv.org/abs/2511.05221. </em></figcaption>
</figure>


---

## Installation

You can install the package from source with
```
conda create -n actitect python=3.9  # use an venv instead if you dont run conda
conda activate actitect
mkdir actitect_experiment
cd actitect_experiment 
git clone git@github.com:db-2010/actitect.git
```
and then simply run 
```
python actitect/install.py       # for the full package, including RBDisco 
or
python actitect/install.py -c    # only the general purpose actigraphy toolkit 
```
*Note: The package might be published to PyPi in the future for a pure pip install.*

after this, you should see
```
... 
2025-10-02 12:34:17 [INFO] Installation successful!
```

---

## ActiTect Usage 

<figure>
  <img src="docs/demo_fig.png" alt="demo actigraphy plot" style="width:500px;">
  <figcaption><em>Figure 2 – Processed actigraphy data. </em></figcaption>
</figure>


You can use the general purpose actigraphy toolkit either as a CLI tool or directly via a light, pythonic API.
The supported actigraphy file formats are
- **Axivity AX3/AX6:** `.cwa`
- **GENEActiv:** `.bin`
- **ActiGraph:** `.gt3x` 
- **Generic:** `.csv` 

<details>
  <summary>Unsupported Devices (Generic CSV)</summary>
   
If your device is not natively supported, you can still use the tool by exporting to a generic `.csv` format first.
The files must contain a column `time` that lists the sampling timestamps in `ISO8601 (%Y-%m-%dT%H:%M:%S.%f%z)` 
format and columns `x, y, z` in units of g:
``` 
time,x,y,z
2023-07-03 15:00:03.699,0.5947265625,0.15283203125,0.742431640625
2023-07-03 15:00:03.707763,0.59033203125,0.16357421875,0.74609375
2023-07-03 15:00:03.715652,0.6279296875,0.136474609375,0.73046875
...
```
</details>

### ActiTect CLI Usage 
If you want to use ActiTect as a CLI tool,you can use
```bash 
actitect-process [-h] [-r ROOT_DIR] [-c CONFIG_FILE] [-d DATA_DIR] [-m META_FILE] [-o OUT_DIR] [-ns] [-cp] [-rp] [-sf]
```
to read and process an entire dataset under a given directory. The full processing includes uniform resampling,
Butterworth bandpass filtering, auto-calibration, non-wear detection, sleep segmentation and 
nocturnal movement feature calculation. 
Please refer to `actitect-process --help` for all options, an overview of the important ones is given below.

<details>
  <summary>File organization options </summary>
  <div id="actitect-file-org"></div>

 - **`-d, --data_dir`** <br> 
 **Description:** (str) The directory (relative to `root_dir` or absolute path) that contains the raw actigraphy files. <br> 
 **Default:** `./data/raw/` <br> 
 **Note:** Only needed if binary files should be located outside the default directory. 
- **`-m, --meta_file`** <br> 
 **Description:** (str) The path to the metadata CSV file, relative to `root_dir` or abs. path. <br> 
 **Default:** `./data/meta/metadata.csv`<br>
- **`-o, --out_dir`** <br>
**Description:** (str) The directory where processed data and features will be stored.<br>
**Default:** ./data/processed/ (relative to `root_dir` or abs. path)<br>
**Note:** By default, both the processed data and calculated features are stored,
 refer to [Operational flags](#actitect-operational-flags) below to view file saving options.<br>
</details>

<details>
  <summary>Operational flags </summary>
  <div id="actitect-operational-flags"></div>

Operational Flags:
- **`-nr, --no_resume`** <br>
**Description:** Disable default resume behavior. By default, ActiTect skips records that already have info-*.json and the expected feature CSVs on disk. <br>
**Default:** resume enabled (flag not set).
- **`-ns, --no_store`** <br>
**Description:** When provided, the script saves only the calculated features and not the processed data.<br>
**Default:** False. <br>
**Note:** If flag is not activated,  the processed actigraphy data is stored as `.parquet` file. 
(~2GB for a 7d recording at 100Hz.)
- **`-cp, --create_plots`** <br>
**Description:** Whether to create plots of the raw and processed actigraphy data.<br>
**Default:** True. <br>
- **`-sf, --skip_feature_calc`** <br>
**Description:**  If enabled, will skip the calculation of motion features.<br>
**Default:** False. <br>
**Note:** Use this if you're only interested in the processing part.
- **`--allow_incomplete_preprocessing`** <br>
**Description:**  If set, allow to continue downstream logic (feature extraction etc.), 
if one of the processing steps has failed.<br>
**Default:** False. <br>
</details>
  
**Note:** You have full flexibility over the steps and parameters (e.g. bandpass 
frequencies, skipping steps etc.), by providing a modified preprocessing config file (`-c`, the default/example 
file can be found [here](./examples/actitect/cli_processing_settings.yaml)).

#### Metadata File and Project Structure
To successfully run the CLI command, you will need a `metadata.csv` file, linking each raw actigraphy file to your
unique identifier, i.e. a patient ID or similar.
<details>
  <summary>metadata.csv format </summary>

The `metadata.csv` file should have one row per subject/record and contain at least the columns 
- ***filename:*** the filename of the actigraphy data.
- ***ID:*** a random identifier for the subject the data is taken from.
- ***record_ID:*** if you have multiple recordings for the same subject, they are processed independently and you can 
use record_ID to assign an identifier to each. E.g. 'left' and 'right' if you have recordings from the both arms.
If one subject (=ID), has multiple records but no entries in record_ID, they will automatically be named rec1, rec2, ... etc. 
- ***diagnosis:*** a string indicating the cohort group of the subject, 
e.g. 'RBD' for RBD diagnosed subject and 'HC' for neg.
- ***train/test:*** should be set to 'test'. (todo: allow to leave empty: implement predict method without eval)
- ***exclude:*** can be used to exclude certain subject from the analysis by setting it to '1', otherwise empty.

| #   | filename                 | ID       | record_ID | diagnosis | train/test | exclude |
|-----|--------------------------|----------|-----------|-----------|------------|---------|
| 1   | `<name1>.<cwa/bin/csv>`  | `<ID-1>` |           | `RBD`     | `test`     |         |
| 2   | `<name2>.<cwa/bin/csv>`  | `<ID-2>` |           | `RBD`     | `test`     | `1`     |
| 3   | `<name3a>.<cwa/bin/csv>` | `<ID-3>` | `left`    | `HC`      | `test`     |         |
| 3   | `<name3b>.<cwa/bin/csv>` | `<ID-3>` | `right`   | `HC`      | `test`     |         |
| ... | ...                      | ...      | ...       | ...       | ...        |

**Note:** if you are only interested in the preprocessing part of the pipeline, the columns ***diagnosis*** and 
***train/test*** are irrelevant and can be left empty. 
</details>

Concerning file organization, you can either specify your desired paths via the [command line options](#actitect-file-org) 
or use the default structure shown below.

<details>
  <summary>Default file organization  </summary>
    <div id="default-file-org"></div>

The actigraphy  binaries are listed under `./data/raw/` and the 
metafile should be stored at `./data/raw/meta/metadata.csv`. 
Processed data will be stored at `./data/processed/` and optionally the RBDisco results under  `./results/`.
```
actitect_experiment/ (the main folder of your experiment)  
  ├── actitect (the cloned repo)
  │     ├── README.md 
  │     ├── libs/
  │     ├── ... 
  ├── data/ 
  │     ├── raw/ (put your actigraphy binaries here...)
  │     │    ├── <name1>.<cwa/bin/csv> 
  │     │    ├── <name2>.<cwa/bin/csv>
  │     │    ├── ...
  │     ├── meta/  (...and don't foget the metadata file.)
  │     │    ├── metadata.csv
  │     ├── processed/ (this is where the results of processing will be stored by default)
  │     │    ├── <ID1>/
  │     │    ├── ...
  │    ...  ...
  ├── results/ (this is where results of RBDisco classifier will be stored by default)
 ... 
```
</details>

### ActiTect API Usage
For users interested in only specific steps of the toolkit, they are accessible via a pythonic API containing 
 - `actitect.api.load()`: load data of any supported device into memory.
 - `actitect.api.process()`: process the data, including preprocessing and non-wear/sleep detection.
 - `actitect.api.plot()`: visualize raw or processed actigraphy data.
 - `actitect.api.compute_per_night_sleep_features()`: compute numerical descriptors of sleep motion patterns.

For a detailed example on how to use each step, see this [example notebook](examples/actitect/actitect_api_example.ipynb).

---

## RBDisco Usage
To use RBDisco to make RBD status predictions for [suited actigraphy files](#actitect-usage-), make sure you
1. Installed the RBDisco extension of ActiTect (see [Installation](#installation))
2. You've run a full processing of your actigraphy files using the [ActiTect CLI](#actitect-cli-usage-),
including the data preprocessing steps and feature extraction (default).

and then run 
```
actitect-rbdisco -d <path>/<to>/<processed>/<data>/<dir> -m <path>/<to>/<meta>/<file>
```
If you did not change the [default paths](#default-file-org) in the [processing CLI](#actitect-cli-usage-), you can omit the 
extra options and simply call `actitect-rbdisco`. Please refer to
`actitect-rbdisco -h` for more options.
The results of the analysis will be stored under `./results/pipeline/run_<date_id>/` and consist of a `.csv` file 
containing individual predictions and a `.json` file containing classification metrics. 

<details>
  <summary>Note: combined CLI  </summary>
    <div id="default-file-org"></div>
  We might provide a combined ActiTect + RBDisco CLI in the future, for now please use the separate CLIs or the API. 

</details>


#### RBDisco API usage
Similar to the core ActiTect API, you can use `actitect.rbdisco.predict()` to use our pretrained models for RBD status 
prediction on any supported actigraphy file. See this [example](examples/rbdisco/rbdisco_api_example.ipynb) for detailed usage.

---

### Note on Reproducibility
The code is seeded, fully deterministic and reproducible. 
However, there are some small platform dependencies linked to the RNG in XGBoost that lead to (small) variations
if executed on different OS like Linux or macOS [
[1](https://github.com/dmlc/xgboost/issues/310),
[2](https://github.com/dmlc/xgboost/issues/3046),
[3](https://github.com/dmlc/xgboost/issues/3523),
[4](https://github.com/dmlc/xgboost/pull/3781),
[5](https://github.com/dmlc/xgboost/issues/9584),
[6](https://discuss.xgboost.ai/t/colsample-by-tree-leads-to-not-reproducible-model-across-machines-mac-os-windows/1709),
[7](https://discuss.xgboost.ai/t/colsample-bytree-seemingly-not-random/106/20)
].
The only way to mitigate this is to match the C++ compilers used in both OS',
e.g. by building XGBoost from source using GCC on macOS:
```
conda activate actitect
brew install gcc@11  
git clone https://github.com/dmlc/xgboost.git
cd xgboost
git checkout 'v2.0.3'   
git submodule update --init --recursive 
mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake ..     
make -j12
cd ../python_package
pip install . 
```
--- 
## Citation
```
@article{bertram2026actitect,
    title = {ActiTect: A Generalizable Machine Learning Pipeline for REM Sleep Behavior Disorder Screening through Standardized Actigraphy},
    author = {David Bertram and Anja Ophey and Sinah R{\"o}ttgen and Konstantin Kufer and Nele Merten and Gereon R. Fink and Elke Kalbe and Clint Hansen and Walter Maetzler and Maximilian Kapsecker and Lara M. Reimer and Stephan Jonas and Andreas T. Damgaard and Natasha B. Bertelsen and Casper Skjaerbaek and Per Borghammer and Karolien Groenewald and Pietro-Luca Ratti and Michele T. Hu and No{\'e}mie Moreau and Michael Sommerauer and Katarzyna Bozek},
    journal = {npj Digital Medicine},
    year = {2026},
    doi = {10.1038/s41746-026-02738-8},
    url = {https://doi.org/10.1038/s41746-026-02738-8}
}
```


<p align="center">
  <a href="https://polyformproject.org/licenses/noncommercial/1.0.0/"><img src="https://img.shields.io/badge/License-PolyForm--NC%201.0.0-blue.svg" alt="License: PolyForm-NC 1.0.0"></a>
  &nbsp;
  <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="CC BY-NC-SA 4.0"></a>
</p>