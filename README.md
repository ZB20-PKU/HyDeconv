# HyDeconv: Hybrid Deconvolution Software

A comprehensive software with an intuitive graphical user interface (GUI) for the resolution enhancement of multiple fluorescence microscopy modalities, including Structured Illumination Microscopy (SIM), Stimulated Emission Depletion microscopy (STED), Multiphoton Microscopy (MPM), Confocal Spinning Disk Microscopy (CSDM), and Wide-Field Microscopy (WFM). 

This repository contains the core source code, pre-trained models, and launch scripts.

## 📦 Repository Contents

*   **`HyDeconv_Handbook.pdf`**: Complete handbook with detailed operation and configuration instructions.
*   **`/src_Hybrid/`**: Core source code directory containing all Python (`.py`) and compiled module (`.pyd`) files.
*   **`/src_Hybrid/src_model/`**: Pre-trained deep learning model files (`.pth`) for different microscopy modalities.
*   **`Launch.py`**: The main script to launch the software graphical user interface (GUI).
*   **`requirements.txt`**: A list of required Python packages.

## ⚡ Quick Start Guide

### 1. Install Anaconda
Download and install Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download).

### 2. Create Conda Environment and Install Dependencies
Open a Command Prompt (or Anaconda Prompt) and run the following commands:
```bash
conda create -n Hybrid_Deconvolution python=3.7 –y
conda activate Hybrid_Deconvolution
conda install cudatoolkit=11.6
conda install git=2.51.0
```
**Important:** Please adjust the versions of cudatoolkit, cupy-cuda, torch, torchvision, and torchaudio to match your specific GPU if necessary.

### 3. Download Software Code
Clone this repository to your local machine and navigate into the software root directory, by running the following commands:
```bash
git clone https://github.com/ZB20-PKU/HyDeconv.git
cd HyDeconv
pip install -r requirements.txt
```

### 4. Download Demonstration Data
The software requires specific training and reconstruction data to function:

Download the `.zip` files from [https://figshare.com/s/c6987ed1e5cd40e9f66c](https://figshare.com/s/c6987ed1e5cd40e9f66c).

Extract them into the software root directory.

**Important:** When extracting the `.zip` files, ensure you select **"Extract Here"** (or equivalent) so that the files and folders are created directly in the software root directory. Do not extract into a new folder named after the `.zip` file.

After extracting correctly, your project folder structure should look like this:
```bash
HyDeconv/
├── HyDeconv_Handbook.pdf
├── requirements.txt
├── Launch.py
├── src_Hybrid/
├── Demo_training_SIM/       # Extracted folder from `Demo_training.zip`
├── Demo_training_STED/      # Extracted folder from `Demo_training.zip`
├── Demo_training_MPM/       # Extracted folder from `Demo_training.zip`
├── Demo_training_CSDM/      # Extracted folder from `Demo_training.zip`
├── Demo_training_WFM/       # Extracted folder from `Demo_training.zip`
├── Demo_Recon_SIM.tif       # Extracted file from `Demo_Recon.zip`
├── Demo_Recon_STED.tif      # Extracted file from `Demo_Recon.zip`
├── Demo_Recon_MPM.tif       # Extracted file from `Demo_Recon.zip`
├── Demo_Recon_CSDM.tif      # Extracted file from `Demo_Recon.zip`
├── Demo_Recon_WFM.tif       # Extracted file from `Demo_Recon.zip`
└── ...
```

### 5. Launch the Software
With the conda environment activated and in the software root directory, run the following command:
```bash
python Launch.py
```
For subsequent launches, run the following commands from the software root directory:
```bash
conda activate Hybrid_Deconvolution
python Launch.py
```

## 📖 Detailed Documentation
For full instructions on software operation and configuration, please refer to the handbook:
[HyDeconv_Handbook.pdf](./HyDeconv_Handbook.pdf)
