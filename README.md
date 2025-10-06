# class-conditional-cp-data-augmentation
Data, trained models, and notebooks for simulations and real-data experiments from the article: 

**Title**: Enhancing Class-Conditional Conformal Prediction for Multiclass Scenarios with Data Augmentation

**Authors**: Andrea Laurenzi, Matteo Borrotti
(submitted to JMLR)

[DOI / arXiv link – add here]


## Repository Structure
- **`dataset/`** – WM-811K wafer map dataset subset and splits
- **`models/`** – Pre-trained models (EfficientNet-B0, CNN)
- **`notebooks/simulations/`** – Jupyter notebooks for empirical experiments (Section 3.1 and Appendix B)
- **`notebooks/real_examples/`** – Jupyter notebooks for empirical experiments (Section 3.2 and Appendix C)
- **`utils/`** – Utility scripts for data processing, model training, data-augmentation and class-conditional conformal predition


## notebooks/nimulations – Empirical part of Section 3.1
This repository contains Jupyter notebooks for the synthetic experiments presented in **Section 3.1** and **Appendix B** of the article.

**Table 5** of the article reports the settings of simulated data scenarios with 8 classes. These experimental settings are provided in the file number_experiments_setup.csv included in this repository.

### Contents
**1. CP_experiments.ipynb**
- Simulates the 39 experiments described in Table 5.
- Trains RF, XGBoost, and MLP models.
- Applies four conformal prediction methods:
  - Standard
  - Classwise
  - Clustered
  - Augmented
- Saves results in CSV files.

**Important Notes:**
- Original runs were executed in Google Colab. Due to disconnection issues, results were split:
  - For **α = 0.10**:  
    `summary_01_part1.csv`, `summary_01_part2.csv`, `summary_01_part3.csv`
  - For **α = 0.05**:  
    `summary_005_part1.csv`, `summary_005_part2.csv`, `summary_005_part3.csv`
- Full simulation can still be executed in one run.
- Run the notebook **twice**:
  - Once for **α = 0.10**
  - Once for **α = 0.05**
  (Set α in the `CFG` class; other parameters are read from `number_experiments_setup.csv`.)

 **2. CP_analysis_01_interval_plots.ipynb** and **3. CP_analysis_02_interval_plots.ipynb**
- Generate the plots reported in **Appendix B** of the article.

## notebooks/real_examples – Empirical part of Section 3.2
This folder contains Jupyter notebooks for the **real-world applications** presented in **Section 3.2** and **Appendix C** of the article:


### Contents

 **1. WMDD_dataset_selection_from_wm811k.ipynb**  
   Constructs a subset of the **WM-811K wafer map dataset** and splits it into **train**, **validation**, **calibration**, and **test** sets.

**2. WM_Effnet_model.ipynb**  
   Trains an **EfficientNet** model using **train + validation** sets and evaluates on **test**.

**3.WM_CNN_Model.ipynb**  
   Trains a **CNN model** using **train + validation** sets and evaluates on **test**.

**4.WM_data_augmentation.ipynb**  
   Applies **traditional data augmentation techniques** to calibration datasets.

**5. WM_effnet_class_conformal.ipynb**  
   Applies **class-conditional conformal prediction** using the trained **EfficientNet model** on **calibration** and **test** sets.  
   

**6. WM_cnn_class_conformal.ipynb**  
   Applies **class-conditional conformal prediction** using the trained **CNN model** on **calibration** and **test** sets.

   **Important Notes:**
- The functions used for conformal predictions in notebooks 5 and 6 are adapted from this [repository](https://github.com/tiffanyding/class-conditional-conformal).

### Workflow overview
<img width="4032" height="1989" alt="image" src="https://github.com/user-attachments/assets/3cd89af9-a336-4645-b82d-d89709d156f1" />
