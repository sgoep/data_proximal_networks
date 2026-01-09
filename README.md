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
