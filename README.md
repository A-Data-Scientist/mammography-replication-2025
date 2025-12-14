# Mammography Replication 2025

This repository contains a replication and extension of the PLOS ONE paper:

> Jaamour A, Myles C, Patel A, Chen SJ, McMillan L, Harris-Birtill D.  
> *A divide and conquer approach to maximize deep learning mammography classification accuracies.* PLOS ONE, 2023.

The original work provides a transparent, open-source pipeline for training and evaluating CNN backbones on the CBIS-DDSM mammography dataset. This fork keeps the same core pipeline and **MobileNetV2** backbone, but adds:

- **Patient-wise, view-aware splitting** for CBIS-DDSM (no patient leakage across splits).
- **Post-hoc probability calibration** via temperature scaling, with Brier, ECE, and reliability plots.
- A **2×2 ablation** over preprocessing (`none` vs `clahe`) and loss (`weighted_ce` vs `focal`).
- An **implementation sweep** mode to reproduce the four main configurations discussed in the 2025 write-up.

The goal is not to invent a new architecture, but to show how evaluation and training choices (splits, calibration, and simple ablations) affect the *clinical reliability* of a lightweight mammography model.

---

## Repository structure (high-level)

- `src/`
  - `main.py` – entry point for training and testing experiments.
  - `cnn_models/` – backbone definitions (e.g., `mobilenet_v2.py`).
  - `data_operations/` – dataset loading, splitting, preprocessing, transforms.
  - `data_visualisation/` – plotting scripts (training curves, confusion matrices, ROC, calibration).
  - `calibration/temperature.py` – temperature scaling (fit, apply, save, load).
  - `config.py` – global configuration (dataset paths, hyperparameters).
  - `utils.py` – utility functions (logging, file saving, etc.).
- `data/`
  - `CBIS-DDSM/` – split manifests and metadata CSVs.
  - `mini-MIAS/` – optional mini-MIAS assets (if used).
- `models/` – SavedModel exports and temperature JSON files.
- `saved_models/` – legacy HDF5 weights (from original code).
- `output/` – training curves and evaluation plots (confusion matrices, ROC).
- `reports/calibration/` – reliability diagrams and calibration reports.
- `debug_preds/` – `.npz` files containing validation labels and probabilities.

---

## Environment setup and usage

### 1. Clone this repository

```bash
git clone https://github.com/A-Data-Scientist/mammography-replication-2025.git
cd mammography-replication-2025
```

### 2. Create a Python environment

Use `conda` or `venv`. The code has been tested with Python 3.8–3.10.

```bash
conda create -n mammography python=3.9
conda activate mammography
```

or

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create output directories

Several scripts expect output locations to exist. From the repo root:

```bash
mkdir -p output
mkdir -p saved_models
mkdir -p models
mkdir -p reports/calibration
mkdir -p debug_preds
```

- `output/` – training curves and comparison plots  
- `saved_models/` – legacy weight dumps from the original project  
- `models/` – SavedModel exports and temperature JSON (`*_temperature.json`)  
- `reports/calibration/` – reliability diagrams + calibration metrics  
- `debug_preds/` – label/probability dumps for analysis

---

## Datasets

### CBIS-DDSM

This project primarily targets **CBIS-DDSM** for benign vs malignant classification.

1. Download the CBIS-DDSM dataset from TCIA.
2. Arrange DICOMs in a directory tree compatible with the paths referenced in `data/CBIS-DDSM` manifests, or set `config.CBIS_ROOT` to point to your local CBIS-DDSM root.
3. Split manifests (`train/val/test` CSV files) are managed under `data/CBIS-DDSM`. This fork adds **patient-wise, view-aware splits** on top of the original file-level splits.

### mini-MIAS (optional)

The original code supports **mini-MIAS** and a **mini-MIAS-binary** task. For the 2025 replication, CBIS-DDSM + MobileNetV2 is the primary focus, but the mini-MIAS code paths remain in place.

See the original project’s instructions or `data/mini-MIAS` for more details on downloading and converting mini-MIAS if you need it.

---

## Running experiments

All commands below assume you are in the `src/` directory:

```bash
cd src
python main.py -h
```

Key arguments:

- `-d, --dataset` – `CBIS-DDSM`, `mini-MIAS`, or `mini-MIAS-binary` (default: `CBIS-DDSM`)
- `-mt, --mammogramtype` – `calc`, `mass`, or `all` (default: `all`)
- `-m, --model` – `VGG-common`, `VGG`, `ResNet`, `Inception`, `DenseNet`, `MobileNet`, `CNN`
- `-r, --runmode` – `train` or `test` (default: `train`)
- `--split_mode` – `classic` (file-level) or `patient` (patient-wise), CBIS-DDSM only
- `--preprocess` – `none` or `clahe`
- `--loss-type` – `weighted_ce` or `focal`
- `--calibrate` – if present, fit and apply **temperature scaling** after training
- `--impl-sweep` – run the four main MobileNetV2 configurations used in the 2025 replication

Other arguments (e.g., `--learning-rate`, `--batchsize`, `--splits`) inherit defaults from the original code. For most reproductions you can leave them unchanged.

---

## 1. Implementation sweep (recommended)

The **implementation sweep** runs the four main configurations discussed in the accompanying report for MobileNetV2 on CBIS-DDSM:

1. Classic file-level split, no calibration  
2. Patient-wise, no calibration  
3. Patient-wise + calibration  
4. Patient-wise + CLAHE + focal loss + calibration  

Run:

```bash
cd src

python main.py \
  -d CBIS-DDSM \
  -mt all \
  -m MobileNet \
  -r train \
  --impl-sweep
```

This is equivalent to sequentially running:

1. `classic, prep=none, loss=weighted_ce, cal=False`  
2. `patient, prep=none, loss=weighted_ce, cal=False`  
3. `patient, prep=none, loss=weighted_ce, cal=True`  
4. `patient, prep=clahe, loss=focal, cal=True`  

For each configuration, the script will:

- Train the MobileNetV2 model with the chosen split / preprocessing / loss.
- Log training and validation accuracy/loss to `output/`.
- Compute validation accuracy (manual + streaming `BinaryAccuracy`).
- Generate a **calibration report**:
  - ROC-AUC, PR-AUC
  - Brier score, ECE, MCE
  - Reliability diagrams saved to `reports/calibration/`.
- Compute confusion matrices and ROC curves via `test_model_evaluation` and save them to `output/`.
- Dump validation labels + probabilities to `debug_preds/*.npz` for offline analysis.

At the end you will see a summary similar to:

```text
=== Validation Accuracy Summary (CBIS-DDSM) ===
classic, prep=none, loss=weighted_ce, cal=False           Acc (uncal): 0.5915
patient, prep=none, loss=weighted_ce, cal=False           Acc (uncal): 0.6300
patient, prep=none, loss=weighted_ce, cal=True            Acc (uncal): 0.6300 | Acc (cal): 0.6300
patient, prep=clahe, loss=focal, cal=True                 Acc (uncal): 0.6645 | Acc (cal): 0.6645
```

Use this sweep to reproduce the **baseline**, **patient-wise**, **calibrated**, and **CLAHE × loss** results and plots from the paper.

---

## 2. Single-run experiments

You can also run individual configurations directly by setting the flags explicitly.

### 2.1 Classic file-level baseline (uncalibrated)

```bash
python main.py \
  -d CBIS-DDSM \
  -mt all \
  -m MobileNet \
  -r train \
  --split_mode classic \
  --preprocess none \
  --loss-type weighted_ce
```

This will:

- Reproduce the original file-level MobileNetV2 baseline.
- Save training curves and confusion matrices + ROC under `output/`.
- Produce an **uncalibrated** calibration report:

  - `reports/calibration/cbis_val_classic_none_weighted_ce_reliability.png`

### 2.2 Patient-wise, view-aware split (uncalibrated)

```bash
python main.py \
  -d CBIS-DDSM \
  -mt all \
  -m MobileNet \
  -r train \
  --split_mode patient \
  --preprocess none \
  --loss-type weighted_ce
```

This configuration:

- Enforces patient-wise, view-aware splits with no calibration.
- Saves patient-wise confusion matrices and ROC to `output/`.
- Generates uncalibrated reliability plots with tag `cbis_val_patient_none_weighted_ce`.

### 2.3 Patient-wise + calibration

```bash
python main.py \
  -d CBIS-DDSM \
  -mt all \
  -m MobileNet \
  -r train \
  --split_mode patient \
  --preprocess none \
  --loss-type weighted_ce \
  --calibrate
```

This configuration:

- Trains as above, then **fits a temperature parameter T** on validation logits.
- Applies T to validation and test outputs.
- Saves T under:

  ```text
  models/MobileNet_CBIS-DDSM*_temperature.json
  ```

- Produces **uncalibrated** and **calibrated** reliability diagrams:

  - `cbis_val_patient_none_weighted_ce_reliability.png` (uncalibrated)
  - `cbis_val_patient_none_weighted_ce_cal_reliability.png` (calibrated)

Accuracy at a 0.5 threshold is unchanged by design; improvements appear in **Brier score, ECE, and reliability curves**.

### 2.4 Patient-wise + CLAHE + focal + calibration (best configuration)

This is the “best” configuration reported in the 2025 replication write-up:

```bash
python main.py \
  -d CBIS-DDSM \
  -mt all \
  -m MobileNet \
  -r train \
  --split_mode patient \
  --preprocess clahe \
  --loss-type focal \
  --calibrate
```

It will:

- Train MobileNetV2 on patient-wise splits using **CLAHE** and **focal loss**.
- Fit and apply temperature scaling.
- Save:
  - Training curves (initial + fine-tuning) under `output/`.
  - Confusion matrices (raw and normalized) and ROC curves for this configuration.
  - Reliability diagrams and calibration metrics under `reports/calibration/`.
  - Temperature parameter and model artifacts under `models/`.

In our experiments, this setup achieved the highest patient-wise test accuracy (~0.6645) among the evaluated combinations.

---

## 3. Test mode (evaluating a saved model)

To evaluate a previously trained model and its calibration on the test subset, use `-r test`. For example, to test the patient-wise + CLAHE + focal configuration:

```bash
python main.py \
  -d CBIS-DDSM \
  -mt all \
  -m MobileNet \
  -r test \
  --split_mode patient \
  --preprocess clahe \
  --loss-type focal
```

In `test` mode:

- The SavedModel is loaded from `models/` using `load_trained_model`.
- The code attempts to load a saved temperature JSON:
  - If found, probabilities are **calibrated** with T.
  - If not, raw probabilities are used.
- Test-set confusion matrices, ROC curves, and calibration reports are written to the same locations as in train mode.

> Note: You do **not** pass `--calibrate` in `test` mode. That flag is only used when fitting T during training.

## Acknowledgements

This work builds directly on the open-source code from the original PLOS ONE paper:

> Jaamour A, Myles C, Patel A, Chen SJ, McMillan L, Harris-Birtill D.  
> *A divide and conquer approach to maximize deep learning mammography classification accuracies.* PLOS ONE, 2023.

It is also informed by evaluation practices from:

- A nationwide AI deployment study in **Nature Medicine** (calibrated, transportable mammography AI).  
- The **MASAI** randomized screening trial in **The Lancet Digital Health** (AI-supported screening with prospectively defined operating points).







----------------------------------------------------------------------------------------------------------------------

Original Repo README









# A Divide and Conquor Approach to Maximise Deep Learning Mammography Classification Accuracies - Published in PLOS ONE [![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg?style=for-the-badge)](https://opensource.org/licenses/BSD-2-Clause) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


**Publication repository of the "_A Divide and Conquor Approach to Maximise Deep Learning Mammography Classification Accuracies_" peer-reviewed paper published in PLOS ONE.** You can read the paper here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0280841

## Abstract

Breast cancer claims 11,400 lives on average every year in the UK, making it one of the deadliest diseases. Mammography is the gold standard for detecting early signs of breast cancer, which can help cure the disease during its early stages. However, incorrect mammography diagnoses are common and may harm patients through unnecessary treatments and operations (or a lack of treatments). Therefore, systems that can learn to detect breast cancer on their own could help reduce the number of incorrect interpretations and missed cases. Various deep learning techniques, which can be used to implement a system that learns how to detect instances of breast cancer in mammograms, are explored throughout this paper.

Convolution Neural Networks (CNNs) are used as part of a pipeline based on deep learning techniques. A divide and conquer approach is followed to analyse the effects on performance and efficiency when utilising diverse deep learning techniques such as varying network architectures (VGG19, ResNet50, InceptionV3, DenseNet121, MobileNetV2), class weights, input sizes, image ratios, pre-processing techniques, transfer learning, dropout rates, and types of mammogram projections.

![CNN Model](https://i.imgur.com/dIfhxyz.png)

Multiple techniques are found to provide accuracy gains relative to a general baseline (VGG19 model using uncropped 512x512 pixels input images with a dropout rate of 0.2 and a learning rate of 1×10^−3) on the Curated Breast Imaging Subset of DDSM (CBIS-DDSM) dataset. These techniques involve transfer learning pre-trained ImagetNet weights to a MobileNetV2 architecture, with pre-trained weights from a binarised version of the mini Mammography Image Analysis Society (mini-MIAS) dataset applied to the fully connected layers of the model, coupled with using weights to alleviate class imbalance, and splitting CBIS-DDSM samples between images of masses and calcifications. Using these techniques, a 5.28% gain in accuracy over the baseline model was accomplished. Other deep learning techniques from the divide and conquer approach, such as larger image sizes, do not yield increased accuracies without the use of image pre-processing techniques such as Gaussian filtering, histogram equalisation and input cropping.

## Citation

### Code citation (this GitHub repository) [![DOI](https://zenodo.org/badge/345135430.svg)](https://zenodo.org/badge/latestdoi/345135430)
```
@software{Jaamour_Adamouization_Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication_PLOS_ONE_2023,
    author = {Jaamour, Adam and Myles, Craig},
    license = {BSD-2-Clause},
    month = may,
    title = {{Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication: PLOS ONE Submission}},
    url = {https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication},
    version = {1.2},
    year = {2023}
}
```

### Published paper citation (PLOS ONE)
```
@article{10.1371/journal.pone.0280841,
    doi = {10.1371/journal.pone.0280841},
    author = {Jaamour, Adam AND Myles, Craig AND Patel, Ashay AND Chen, Shuen-Jen AND McMillan, Lewis AND Harris-Birtill, David},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {A divide and conquer approach to maximise deep learning mammography classification accuracies},
    year = {2023},
    month = {05},
    volume = {18},
    url = {https://doi.org/10.1371/journal.pone.0280841},
    pages = {1-24},
    number = {5},
}
```

## Environment setup and usage

Clone the repository:

```
git clone https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication
cd Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication
```

Create a virtual conda environment:

```
conda create -n mammography python=3.6.13
conda activate mammography
```

Install requirements:
```
pip install -r requirements.txt
```

Create `output`and `save_models` directories to store the results:

```
mkdir output
mkdir saved_models
```

`cd` into the `src` directory and run the code:

```
cd ./src
main.py [-h] -d DATASET [-mt MAMMOGRAMTYPE] -m MODEL [-r RUNMODE] [-lr LEARNING_RATE] [-b BATCHSIZE] [-e1 MAX_EPOCH_FROZEN] [-e2 MAX_EPOCH_UNFROZEN] [-roi] [-v] [-n NAME]
```

where:
* `-h` is a flag for help on how to run the code.
* `DATASET` is the dataset to use. Must be either `mini-MIAS`, `mini-MIAS-binary` or `CBIS-DDMS`. Defaults to `CBIS-DDMS`.
* `MAMMOGRAMTYPE` is the type of mammograms to use. Can be either `calc`, `mass` or `all`. Defaults to `all`.
* `MODEL` is the model to use. Must be either `VGG-common`, `VGG`, `ResNet`, `Inception`, `DenseNet`, `MobileNet` or `CNN`.
* `RUNMODE` is the mode to run in (`train` or `test`). Default value is `train`.
* `LEARNING_RATE` is the optimiser's initial learning rate when training the model during the first training phase (frozen layers). Defaults to `0.001`. Must be a positive float.
* `BATCHSIZE` is the batch size to use when training the model. Defaults to `2`. Must be a positive integer.
* `MAX_EPOCH_FROZEN` is the maximum number of epochs in the first training phrase (with frozen layers). Defaults to `100`.
* `MAX_EPOCH_UNFROZEN`is the maximum number of epochs in the second training phrase (with unfrozen layers). Defaults to `50`.
* `-roi` is a flag to use versions of the images cropped around the ROI. Only usable with mini-MIAS dataset. Defaults to `False`.
* `-v` is a flag controlling verbose mode, which prints additional statements for debugging purposes.
* `NAME` is name of the experiment being tested (used for saving plots and model weights). Defaults to an empty string.

## For best model described in paper:
```
python main.py -d CBIS-DDSM -mt all -m MobileNet -r train -lr 0.0001
python main.py -d CBIS-DDSM -mt all -m MobileNet -r test -lr 0.0001
```

## Dataset installation

#### DDSM and CBIS-DDSM datasets

These datasets are very large (exceeding 160GB) and more complex than the mini-MIAS dataset to use. They were downloaded by the University of St Andrews School of Computer Science computing officers onto \textit{BigTMP}, a 15TB filesystem that is mounted on the Centos 7 computer lab clients with NVIDIA GPUsusually used for storing large working data sets. Therefore, the download process of these datasets will not be covered in these instructions.

The generated CSV files to use these datasets can be found in the `/data/CBIS-DDSM` directory, but the mammograms will have to be downloaded separately. The DDSM dataset can be downloaded [here](http://www.eng.usf.edu/cvprg/Mammography/Database.html), while the CBIS-DDSM dataset can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#5e40bd1f79d64f04b40cac57ceca9272).

#### mini-MIAS dataset

* This example will use the [mini-MIAS](http://peipa.essex.ac.uk/info/mias.html) dataset. After cloning the project, travel to the `data/mini-MIAS` directory (there should be 3 files in it).

* Create `images_original` and `images_processed` directories in this directory: 

```
cd data/mini-MIAS/
mkdir images_original
mkdir images_processed
```

* Move to the `images_original` directory and download the raw un-processed images:

```
cd images_original
wget http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
```

* Unzip the dataset then delete all non-image files:

```
tar xvzf all-mias.tar.gz
rm -rf *.txt 
rm -rf README 
```

* Move back up one level and move to the `images_processed` directory. Create 3 new directories there (`benign_cases`, `malignant_cases` and `normal_cases`):

```
cd ../images_processed
mkdir benign_cases
mkdir malignant_cases
mkdir normal_cases
```

* Now run the python script for processing the dataset and render it usable with Tensorflow and Keras:

```
python3 ../../../src/dataset_processing_scripts/mini-MIAS-initial-pre-processing.py
```

## License 
* see [BSD 2-Clause License](https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication/blob/master/LICENSE) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning,Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication&type=Date)](https://star-history.com/#Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning&Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication&Date)

## Authors

* [Adam Jaamour](https://orcid.org/0000-0002-8298-1302) (adam[at]jaamour[dot]com)
* [Craig Myles](https://orcid.org/0000-0002-2701-3149)
* Ashay Patel
* Shuen-Jen Chen
* [Lewis McMillan](https://orcid.org/0000-0002-7725-5162)
* [David Harris-Birtill](https://orcid.org/0000-0002-0740-3668)
