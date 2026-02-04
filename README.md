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

**Important:** Please ensure **Anaconda is added to your system's PATH environment variable** during installation.

### 2. Create Conda Environment
Open the Command Prompt and run the following commands to create environment:
```bash
conda create -n Hybrid_Deconvolution python=3.7 –y
conda activate Hybrid_Deconvolution
conda install git
```
**Important:** Please adjust the versions of cudatoolkit, cupy-cuda, torch, torchvision, and torchaudio to match your specific GPU if necessary.

### 3. Download Software Code and Install Dependencies
Run the following commands to clone the repository and install all required dependencies:
```bash
git clone https://github.com/ZB20-PKU/HyDeconv.git
cd HyDeconv
pip install -r requirements.txt
conda install cudatoolkit=11.6 -c conda-forge
```
**Important:** If any package during the `pip install` process reports an error like "Microsoft Visual C++ 14.0 or greater is required", you will need to install the **Microsoft C++ Build Tools**:
1. Visit the [Visual Studio Downloads page](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Download and run the "Visual Studio Build Tools" installer.
3. In the workload selection interface, check the box for "Desktop development with C++" and proceed with the installation.
4. A system restart may be required. Then, re-run the `pip install` command above.

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
For the first launch, run the following command:
```bash
python Launch.py
```
For subsequent launches, run the following commands:
```bash
conda activate Hybrid_Deconvolution
cd HyDeconv
python Launch.py
```

## 📖 Detailed Documentation
For full instructions on software operation and configuration, please refer to the handbook:
[HyDeconv_Handbook.pdf](./HyDeconv_Handbook.pdf)
