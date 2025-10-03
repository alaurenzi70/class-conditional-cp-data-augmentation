# ====================================================
# CFG FOR CNN
# ====================================================


epochs = 200  # Reduced from 500, sufficient for CNN on 101x101 images
patience = 20  # More balanced value to stop earlier in case of overfitting
normalize = True  
model_save = True  # Automatically save the best model
num_classes = 8  
input_shape = (101, 101, 1)  
batch_size = 64  # Larger batch size if possible, to speed up training
output_dir = '/content/drive/MyDrive/Conformal_Prediction_Research/Class_Conditional_CP_with_Data_Augmentation/WMDD_application/models'
#directory to save the model 