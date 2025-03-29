# Radar-Camera Fusion with MATLAB & PyTorch

This repository provides a framework for integrating MATLAB and PyTorch for radar-camera fusion. It includes data preprocessing, model scripting, and Simulink integration.

## 📌 Prerequisites

Ensure you have the following installed:

- **MATLAB** (with Deep Learning Toolbox)
- **Python** (tested with 3.10.8)
- **PyTorch** (tested with 1.13.0+cpu)
- **NumPy** (tested with 1.23.4)

## ⚙️ Setup

### 1️⃣ Python Virtual Environment Setup

Run the following commands in a Windows terminal:

```sh
python -m venv env
env\Scripts\activate
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.23.4
