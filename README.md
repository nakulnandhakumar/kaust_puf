# KAUST_PUF
## Chaotic-VCSEL PUF System with AI-Based Authentication Toolkit

This repository provides a complete toolkit for authentication based on the chaotic output of D-shaped Vertical-Cavity Surface-Emitting Lasers (VCSELs), as presented in our project:
> **Chaos-based scalable optoelectronic physical unclonable functions with AI-driven dynamic authentication**  
> [ResearchSquare Preprint](https://www.researchsquare.com/article/rs-6484421/v1)

Each laser, under fixed physical bias (current, temperature, and device ID), produces a unique time-series intensity pattern that serves as a challenge-response pair (CRP). These physical signatures are classified using deep learning, tested against adversarial attacks, and evaluated through information-theoretic and dynamical metrics.


## Quick Start

1. Place the CSV data of chaotic-VCSEL CRPs into a `data/` folder.  
   > *Note: Access to the original CRP dataset is subject to availability and research context. Please contact the authors.* 

2. Train or evaluate the CNN classifier using `PUF_Classifier.py`, or load a model from `saved_models/`.  
3. Use `VAE.py` to generate synthetic sequences for adversarial training.  
4. Evaluate entropy, Lyapunov exponent, correlation, and Hamming distance via `statistical_evaluation/`.  
5. Simulate black-box and gray-box adversarial attacks using `adversary_emulation/`.  
6. Assess long-term stability and fine-grained challenge separability using `stability_spearability/`.  
7. Visualize key evolution over time using `QRcode.py`.  
8. For large-scale experiments, use job scripts under `jobscripts/` for KAUST IBEX.


## Structure

### **1. `PUF_Classifier.py`**  
- Implements a 1D Convolutional Neural Network (CNN) to:  
  - Classify CRP sequences and identify the source VCSEL.  
  - Load and process time-series intensity data from CSV files.

### **2. `VAE.py`**  
- Contains a Variational Autoencoder for:  
  - Learning generative models of chaotic sequences.  
  - Creating synthetic adversarial inputs to enhance CNN robustness.

### **3. `adversary_emulation/`**  
- Simulates common attack scenarios using:  
  - `Blackbox_Attack.py`: Mimics a setting where the attacker has no access to model internals.  
  - `Graybox_Attack.py`: Assumes partial knowledge of model structure or parameters.

### **4. `saved_models/`**  
- Contains pre-trained CNN weights:  
  - `Original_CNN.pth`: Baseline model trained on raw data.  
  - `Enhanced_CNN.pth`: Model hardened via adversarial training.

### **5. `statistical_evaluation/`**  
- Provides scripts to quantify key metrics from chaotic sequences:  
  - `Calculation_Hmin_LE.py`: Minimum entropy and Lyapunov exponent.  
  - `Calculation_HD.py`: Fractional Hamming distance.  
  - `corr_coeff.py`: Pearson correlation coefficients.

### **6. `stability_spearability/`**  
- Evaluates classifier performance under deployment-relevant conditions:  
  - `Model_for_system_stability.py`: Assesses long-term operating stability.  
  - `Model_for_challenge_separability.py`: Measures ability to resolve closely spaced challenges after incremental retraining.

### **7. `QRcode.py`**  
- Encodes segments of chaotic sequences into standardized QR codes.  
- Demonstrates the system's ability to continuously generate scannable keys over time.

### **8. `jobscripts/`**  
- Shell scripts for distributed training and evaluation on the KAUST IBEX supercomputing platform.



## Dependencies

This repository relies on several Python libraries for data processing, visualization, and machine learning. Make sure to install the following dependencies before running the code:

| Dependency    | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `numpy`       | Fundamental package for numerical computations and array manipulation.     |
| `matplotlib`  | Library for creating static, interactive, and animated visualizations.     |
| `pandas`      | Data manipulation and analysis tool for handling structured datasets.      |
| `pytorch`     | Deep learning framework for building and training neural networks.         |
| `scipy`       | Scientific computing library with modules for optimization, integration, and statistics. |
| `seaborn`     | Statistical data visualization library built on top of matplotlib.         |

**Installation:** It is recommended to set up a virtual environment and conduct all project work inside. Install all required dependencies inside virtual environment via `pip`:
```bash
pip install numpy matplotlib pandas torch scipy seaborn
```


## Citation

If you find this project useful in your research, please consider citing our work:

> Zhican Zhou, Hang Lu, Nakul Nandhakumar et al.  *21 April 2025, PREPRINT (Version 1), Research Square*.  
> [https://doi.org/10.21203/rs.3.rs-6484421/v1](https://doi.org/10.21203/rs.3.rs-6484421/v1)

## License  
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

