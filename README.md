⚠️ **Note:** This repository is currently under active development. Expect potential instability.


# Data-Proximal Neural Networks for Limited-View CT

This repository contains the code accompanying the poster **"Data-Proximal Neural Networks for Limited-View CT"** presented at BVM 2025. The project implements and evaluates a family of neural network architectures designed for consistent and artifact-reduced image reconstruction from limited-angle CT measurements.

## 🧠 Summary

Limited-angle computed tomography (CT) is an ill-posed inverse problem. Conventional data-driven methods offer strong performance but often lack guarantees of data consistency. This project addresses this issue using **Data-Proximal (DP) Neural Networks**, which:

- Generalize residual and null-space networks.
- Ensure that network outputs remain close to the measured data.
- Are backed by a rigorous mathematical regularization framework.

## 📖 Reference

> Göppel, S., Frikel, J., & Haltmeier, M.  
> *Data-proximal null-space networks for inverse problems*  
> [arXiv:2309.06573](https://arxiv.org/abs/2309.06573)

## 🏗️ Architecture

The repository implements the following models:

- **Residual Network (RS)**  
  $$N_θ^{RS}(f) = f + U_θ(f)$$

- **Null-Space Network (NS)**  
  $$N_θ^{NS}(f) = f + (Id - A_I^+ A_I) U_θ(f)$$

- **Data-Proximal Network (DP)**  
  $$N_θ^{DP}(f) = f + (Id - A_I^+ A_I) U_θ(f) + A_I^+ Φ_β(A_I U_θ(f))$$

where $A_I $ is the limited-angle Radon transform and $Φ_β$ enforces data-proximity.

## 🧪 Experiments

We evaluate all models on:

- **Synthetic Shepp-Logan phantoms**
- **LoDoPaB-CT dataset** (low-dose clinical CT data)

Evaluation metrics:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)


## ▶️ Getting Started

### 1. Clone and setup

```bash
git clone git@github.com:sgoep/data_proximal_networks.git
cd data_proximal_networks
conda env create -f environment.yml
conda activate data_prox2
```

### 2. Download synthetic data

You can download the files for synthetic data from
https://drive.google.com/drive/folders/1JJuDGj35XQEeDdDCUMgIuADHqThnNDf6?usp=share_link

Put it into
```bash
data/data_synthetic/
```
afterwards.

### 3. Define used network architectures

Network architectures used for training and testing are defined in `example.py`. The name contains the initial reconstruction, i.e. Landweber iteration, TV or sparse $\ell_1$ regularization, followed by a residual, null-space or data-proximal null-space network, respectively.

To remove experiments from the pipeline, comment or remove lines from this list:
```bash
models = [
    "landweber_res",
    "tv_res",
    "ell1_res",
    "landweber_nsn",
    "tv_nsn",
    "ell1_nsn",
    "landweber_dp_nsn",
    "tv_dp_nsn",
    "ell1_dp_nsn",
]
```

### 4. Train the network

```bash
python -m src.models.training synthetic
```

### 5. Evaluate

```bash
python -m src.models.testing synthetic
```