# Required libraries
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

# Generate the full path to an image based on the dataset split and image name
def image_path_generation(row, base_path='/wmdd811k/images'):
    """
    Generate the full path to an image based on the dataset split and image name.

    Parameters:
    - row (pd.Series): A row with at least 'set' and 'image' fields.
    - base_path (str): The base directory containing the image folders.

    Returns:
    - str: Full path to the image file.
    """
    subfolder = row['set']
    return os.path.join(base_path, subfolder, row['image'])


# 
# Valid sets shared across the codebase
VALID_SETS = {"train", "val", "test", "cal"}

# Function to generate the image path based on augmented dataset
def image_path_generation_aug(row, base_path='/wmdd811k/images'):
    """
    Generates the full image path for a given row.

    Parameters:
        row (dict or pd.Series): Contains at least 'set' and 'image' keys.
        base_path (str): Base directory where images are stored.

    Returns:
        str: Full path to the image.

    Raises:
        ValueError: If 'set' is not one of the valid options.
    """
    if row["set"] not in VALID_SETS:
        raise ValueError(f"Invalid set name: {row['set']}. Expected one of {VALID_SETS}.")
    return os.path.join(base_path, row["image"])


# Function to get model predictions (softmax probabilities) from a dataloader
def get_model_probs(model, dataloader, device):
    model.eval()
    probs = []
    for images in tqdm(dataloader, total=len(dataloader)):
        images = images[0].to(device)  # Adjust if your DataLoader returns a tuple
        with torch.no_grad():
            outputs = model(images)
            outputs = outputs.softmax(dim=1).cpu().numpy()
            probs.append(outputs)
    return np.concatenate(probs)

# Function to compute and display evaluation metrics
def evaluate_model(y_true, y_pred, y_probs):
    # Compute macro-averaged recall (average recall across all classes)
    accuracy = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro')
   
    print(f'Overall Accuracy: {accuracy:.3f}')
    print(f'Macro-averaged recall score: {macro_recall:.3f}')
    

# Compute ROC AUC score per class in a one-vs-all manner
def roc_auc_score_multiclass(actual_class, pred_class):
    """
    Compute the ROC AUC score for each class in a one-vs-all fashion.

    Parameters:
    - actual_class (list or array): True class labels
    - pred_class (list or array): Predicted class labels

    Returns:
    - dict: ROC AUC scores for each class
    """
    unique_classes = set(actual_class)
    roc_auc_dict = {}
    for cls in unique_classes:
        binary_actual = [1 if x == cls else 0 for x in actual_class]
        binary_pred = [1 if x == cls else 0 for x in pred_class]
        try:
            roc_auc = roc_auc_score(binary_actual, binary_pred)
        except ValueError:
            roc_auc = float('nan')  # Handle case with no positive samples
        roc_auc_dict[cls] = roc_auc
    return roc_auc_dict

# Plot the label distribution as a bar chart
def plot_label_distribution(df, label_col='labels', title='Label Distribution',
                            figsize=(6, 4), color_palette='Set2'):
    """
    Plot a bar chart of label distribution with annotations and full grid.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the label column.
    - label_col (str): Name of the column containing the labels.
    - title (str): Title of the plot.
    - figsize (tuple): Size of the figure (width, height).
    - color_palette (str or list): Color palette for the bars.
    """
    counts = df[label_col].value_counts().sort_values(ascending=False)

    plt.figure(figsize=figsize)
    ax = plt.bar(counts.index.astype(str), counts.values, color=sns.color_palette(color_palette, len(counts)))

    # Annotate each bar with its count
    for index, value in enumerate(counts.values):
        plt.text(index, value + (0.01 * counts.max()), f'{value}',
                 ha='center', va='bottom', fontsize=8)

    plt.title(title, fontsize=13)
    plt.xlabel('Labels', fontsize=10)
    plt.ylabel('Counts', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.8)  # Full grid
    plt.tight_layout()
    plt.show()

# Conformal
# Reset index and extract labels and prediction probabilities
def prepare_data(dataset, probs):
    dataset.reset_index(drop=True, inplace=True)  # Ensure a clean index after any filtering or concatenation
    labels = dataset.labels.values                # Extract true labels
    smx = probs                                   # Extract model output probabilities
    return labels, smx

# Compute APS scores (Adaptive Prediction Sets)
def get_scores(smx):
    return get_APS_scores_all(smx, randomize=True, seed=0)  # Compute APS scores with fixed seed for reproducibility