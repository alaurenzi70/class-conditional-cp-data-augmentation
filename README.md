# class-conditional-cp-data-augmentation
Data, trained models, and notebooks for simulations and real-data experiments from the article: 

**Title**: Enhancing Class-Conditional Conformal Prediction for Multiclass Scenarios with Data Augmentation

**Authors**: Andrea Laurenzi, Matteo Borrotti
(submitted to JMLR)

[DOI / arXiv link – add here]


## Repository Structure
- **`dataset/`** – WM-811K wafer map dataset subset and splits
- **`models/`** – Pre-trained models (EfficientNet-B0, CNN)
- **`notebooks/simulations/`** – Jupyter notebooks for empirical experiments (Section 3.2 and Appendix C)
- **`notebooks/real_examples/`** – Jupyter notebooks for empirical experiments (Section 3.2 and Appendix C)
- **`utils/`** – Utility scripts for data processing, model training, data-augmentation and class-conditional conformal predition


## Notebooks/Real Examples – Empirical Part of Section 3
This folder contains Jupyter notebooks for the **real-world applications** presented in **Section 3.2** and **Appendix C** of the article:


### Contents

1. **WMDD Dataset Selection from WM811k**  
   Constructs a subset of the **WM-811K wafer map dataset** and splits it into **train**, **validation**, **calibration**, and **test** sets.

2. **WM_Effnet_model**  
   Trains an **EfficientNet** model using **train + validation** sets and evaluates on **test**.

3. **WM_CNN_model**  
   Trains a **CNN model** using **train + validation** sets and evaluates on **test**.

4. **WM_data_augmentation**  
   Applies **traditional data augmentation techniques** to calibration datasets.

5. **WM_effnet_class_conformal**  
   Applies **class-conditional conformal prediction** using the trained **EfficientNet model** on **calibration** and **test** sets.  
   The functions used for conformal predictions are adapted from this [repository](https://github.com/tiffanyding/class-conditional-conformal).

6. **WM_cnn_class_conformal**  
   Applies **class-conditional conformal prediction** using the trained **CNN model** on **calibration** and **test** sets.

## Workflow overview
<img width="4032" height="1989" alt="image" src="https://github.com/user-attachments/assets/3cd89af9-a336-4645-b82d-d89709d156f1" />
