# ====================================================
# CFG
# ====================================================


print_freq=100
num_workers = 18
model_name = 'tf_efficientnet_b0_ns' #'vgg16' #'resnext50_32x4d' #'tf_efficientnet_l2_ns_475'  #'tf_efficientnet_b2_ns' #'resnext50_32x4d'  #'coat_tiny'
output_dir = '/content/drive/MyDrive/Conformal_Prediction_Research/Class_Conditional_CP_with_Data_Augmentation/WMDD_application/models' 
intervalplot_dir = '/content/drive/MyDrive/Conformal_Prediction_Research/Class_Conditional_CP_with_Data_Augmentation/WMDD_application/resuls'
#model_dir='./'
size = 101 #712
epochs = 100
factor = 0.2
patience_lr = 3
patience_es = 5
eps = 1e-6
lr = 1e-4
min_lr = 1e-6
batch_size = 16
weight_decay = 1e-6
gradient_accumulation_steps = 1
max_grad_norm = 1000
seed = 42
target_size = 8
target_col = 'labels'
n_fold = 3
#trn_fold = [1,2,3,4,5]
trn_fold = [0,1,2]
score_plot = True
label2int = {
    'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 
    'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7
}
alpha = 0.01
score = 'APS'  # method to compute scores ('APS', 'RAPS', 'STD')