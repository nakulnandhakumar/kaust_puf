# KAUST_PUF
## Chaotic-VCSEL PUF System with AI-Based Authentication Toolkit

This repository provides a complete toolkit for authentication based on the chaotic output of D-shaped Vertical-Cavity Surface-Emitting Lasers (VCSELs), as presented in our article:
> **Physical unclonable functions based on chaotic vertical-cavity surface-emitting lasers for dynamic authentication**  
> [Nature Electronics](https://www.nature.com/articles/s41928-026-01627-y)

Each laser, under fixed physical bias (current, temperature, and device ID), produces a unique time-series intensity pattern that serves as a challenge-response pair (CRP). These physical signatures are classified using deep learning, tested against adversarial attacks, and evaluated through information-theoretic and dynamical metrics.


## Quick Start

1. Prepare the chaos-based CRP inputs according to the local data organization, and update the demo placeholder filenames in the scripts as needed.  
2. Train or evaluate the CNN classifier using `puf_classifier_save_model.py`, or load a model from `saved_models/`.  
3. Use `VAE.py` to generate synthetic sequences for adversarial training.  
4. Evaluate entropy, Lyapunov exponent, correlation, and auxiliary key-level Hamming distance metrics via `statistical_evaluation/`.  
5. Simulate black-box and gray-box adversarial attacks using `adversary_emulation/`.  
6. Assess long-term stability and fine-grained challenge separability using `stability_spearability/`.  
7. Visualize key evolution over time using `QRcode.py`.  

Set `num_classes = N` consistently across CNN training, saved CNN weights, VAE loading, and black-box/gray-box attack scripts, where `N` is the number of enrolled PUF classes in the local dataset.


## Structure

### **1. `puf_classifier_save_model.py`**  
- Implements a 1D Convolutional Neural Network (CNN) to:  
  - Classify CRP sequences and identify the source VCSEL.  
  - Load and process time-series intensity data from CSV files.

### **2. `VAE.py`**  
- Contains a Variational Autoencoder for:  
  - Learning generative models of chaotic sequences.  
  - Creating synthetic adversarial inputs to enhance CNN robustness.

### **3. `adversary_emulation/`**  
- Simulates common attack scenarios using:  
  - `Blackbox_attack.py`: Mimics a setting where the attacker has no access to model internals.  
  - `Graybox_attack.py`: Assumes partial knowledge of model structure or parameters.

### **4. `saved_models/`**  
- Stores locally generated CNN weights with an `N`-class output head:  
  - `CNN_N_classes.pth`: Baseline model trained on raw data.  
  - `Enhanced_CNN_N_classes.pth`: Model hardened via adversarial training.
- The public demo does not require disclosing the dataset-specific value of `N`; keep the same `N` when saving and loading models.

### **5. `statistical_evaluation/`**  
- Provides scripts to quantify key metrics from chaotic sequences:  
  - `Calculation_Hmin_LE.py`: Minimum entropy and Lyapunov exponent.  
  - `Calculation_HD.py`: Fractional Hamming distance for within-key segment comparisons and between-key comparisons; this is an auxiliary bit-level statistic rather than the reproducibility/uniqueness criterion used for authentication.  
  - `Corr_coeff.py`: Pearson correlation coefficients.
- Demo statistic inputs should be prepared as one-column numeric sequence CSV files; any mapping from local raw-data columns to that sequence format is kept outside the public demo code.

### **6. `stability_spearability/`**  
- Evaluates classifier performance under deployment-relevant conditions:  
  - `Model_for_system_stability.py`: Assesses long-term operating stability.  
  - `Model_for_challenge_separability.py`: Measures ability to resolve closely spaced challenges after incremental retraining.

### **7. `QRcode.py`**  
- Encodes segments of chaotic sequences into standardized QR codes.  
- Demonstrates the system's ability to continuously generate scannable keys over time.




## Dependencies

This repository relies on several Python libraries for data processing, visualization, and machine learning. Make sure to install the following dependencies before running the code:

| Dependency    | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `numpy`       | Fundamental package for numerical computations and array manipulation.     |
| `pandas`      | Data manipulation and analysis tool for handling structured datasets.      |
| `pytorch`     | Deep learning framework for building and training neural networks.         |
| `scipy`       | Scientific computing library with modules for optimization, integration, and statistics. |
| `scikit-learn`| Logistic regression and cross-validation utilities for separability analysis. |
| `joblib`      | Runtime dependency used by scikit-learn in some Python environments.       |
| `qrcode`      | QR code generation for key visualization demos.                            |

**Installation:** It is recommended to set up a virtual environment and conduct all project work inside. Install all required dependencies inside virtual environment via `pip`:
```bash
pip install numpy pandas torch scipy scikit-learn joblib "qrcode[pil]"
```


## Citation

If you find this project useful in your research, please consider citing our work:

> Zhou, Z., Lu, H., Nandhakumar, N. et al. Physical unclonable functions based on chaotic vertical-cavity surface-emitting lasers for dynamic authentication. *Nat Electron* (2026).  
> [https://doi.org/10.1038/s41928-026-01627-y](https://doi.org/10.1038/s41928-026-01627-y)

## License  
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

