# Hybrid Deconvolution Software

A comprehensive software package for advanced deconvolution and reconstruction of microscopy images (e.g., SIM, STED, WFM, MPM, CSDM). This repository contains the core source code, pre-trained models, and launch scripts.

## 📦 Repository Contents

*   **`/src_Hybrid/`**: Core source code directory containing all Python (`.py`) and compiled module (`.pyd`) files for various reconstruction algorithms (Hessian, Sparse, TDV, etc.), optical models, and utilities.
*   **`/src_Hybrid/src_model/`**: Pre-trained model files (`.pth`) for different microscopy modalities.
*   **`Launch.py`**: The main script to launch the software interface.
*   **`requirements.txt`**: List of Python dependencies.
*   **`Hybrid Deconvolution Software Handbook.pdf`**: Complete user manual with detailed installation, configuration, and usage instructions.


## ⚡ Quick Start Guide

### 1. Get Software Code
Clone this repository to your local machine and navigate into the project directory:
```bash
git clone https://github.com/ZB20-PKU/Hybrid-Deconvolution-Software.git
cd Hybrid-Deconvolution-Software
```

### 2. Download Demonstration Data
The software requires specific training and reconstruction data to function:

Download the `.zip` files from the figshare link below.

Extract them into the root directory of this cloned project.

**Data Link:** [https://figshare.com/s/c6987ed1e5cd40e9f66c](https://figshare.com/s/c6987ed1e5cd40e9f66c)

**Important:** When extracting the `.zip` files, ensure you select **"Extract Here"** (or equivalent) so that the files and folders are created directly in the project root. Do not extract into a new folder named after the `.zip` file.

After extracting correctly, your project folder structure should look like this:
```bash
Hybrid-Deconvolution-Software/
├── Hybrid Deconvolution Software Handbook.pdf
├── Launch.py
├── README.md
├── src_Hybrid/
├── Demo_training_SIM/       # Extracted from `Demo_training.zip`
├── Demo_training_STED/      # Extracted from `Demo_training.zip`
├── Demo_training_MPM/       # Extracted from `Demo_training.zip`
├── Demo_training_CSDM/      # Extracted from `Demo_training.zip`
├── Demo_training_WFM/       # Extracted from `Demo_training.zip`
├── Demo_Recon_SIM.tif       # Extracted from `Demo_Recon.zip`
├── Demo_Recon_STED.tif      # Extracted from `Demo_Recon.zip`
├── Demo_Recon_MPM.tif       # Extracted from `Demo_Recon.zip`
├── Demo_Recon_CSDM.tif      # Extracted from `Demo_Recon.zip`
├── Demo_Recon_WFM.tif       # Extracted from `Demo_Recon.zip`
└── ...
```

## 📖 Detailed Documentation
For full instructions on software operation, parameter configuration, and advanced features, please refer to the user manual:
[Hybrid Deconvolution Software Handbook.pdf](./Hybrid%20Deconvolution%20Software%20Handbook.pdf)
