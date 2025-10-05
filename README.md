# class-conditional-cp-data-augmentation
Data, trained models, and notebooks for simulations and real-data experiments from the paper “Class-Conditional Conformal Prediction with Data Augmentation” (submitted to JMLR)


# Notebooks/Real Examples – Empirical Part of Section 3
This folder contains Jupyter notebooks for the **real-world applications** presented in **Section 3.2** and **Appendix C** of the article:


## Contents

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
   Functions adapted from this [repository](https://github.com/tiffanyding/class-conditional-conformal)..

6. **WM_cnn_class_conformal**  
   Applies **class-conditional conformal prediction** using the trained **CNN model** on **calibration** and **test** sets.  

