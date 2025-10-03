# Required libraries 
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine
from torchvision.io import read_image, ImageReadMode
from sklearn.model_selection import train_test_split

# Image augmentation function ---

def augment_and_save_image(image_path, transform, save_path):
    """
    Apply transformations to an image and save the result.

    Parameters:
    - image_path (str): Path to the original image.
    - transform (callable): Transformation to apply.
    - save_path (str): Path to save the augmented image.
    """
    image = read_image(image_path, mode=ImageReadMode.RGB)  # Load image as tensor
    augmented_image = transform(image)                      # Apply transformation
    augmented_image = torchvision.transforms.ToPILImage()(augmented_image)  # Convert to PIL
    augmented_image.save(save_path)                         # Save the image

# Class balancing and augmentation function 

def balance_classes_and_augment(df, target_count, save_dir, transform, output_path):
    """
    Balance dataset by oversampling underrepresented classes via data augmentation.

    Parameters:
    - df (pd.DataFrame): Original DataFrame with 'image_id' and 'labels' columns.
    - target_count (int): Target number of images per class.
    - save_dir (str): Directory to save augmented images.
    - transform (callable): Transformations to apply to images.
    - output_path (str): Path to save the updated DataFrame.
    """
    class_counts = df['labels'].value_counts().to_dict()
    new_rows = []
    saved_image_count = 0

    # Filter classes with fewer images than the target
    filtered_counts_class = {key: val for key, val in class_counts.items() if val < target_count}
    
    if not filtered_counts_class:
        print("All classes are already balanced or no minority class found.")
        return df
    
    keys_list = list(filtered_counts_class.keys())
    counts = list(filtered_counts_class.values())

    for i in tqdm(range(len(keys_list)), desc="Processing Classes"):
        num_to_augment = target_count - counts[i]
        class_images = df[df['labels'] == keys_list[i]]
        
        for j in tqdm(range(num_to_augment), desc=f"Augmenting {keys_list[i]}", leave=False):
            row = class_images.sample(n=1)
            image_id = row['image_id'].iloc[0]
            labels = row['labels'].iloc[0]

            original_name = os.path.basename(image_id)
            name, ext = os.path.splitext(original_name)
            new_name = f"{name}_{j}{ext}"
            new_image_id = os.path.join(save_dir, new_name)
            
            # Apply augmentation and save image
            augment_and_save_image(image_id, transform, new_image_id)
            saved_image_count += 1

            new_rows.append({'image_id': new_image_id, 'labels': labels})

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_path, index=False)
    
    print(f"Total new images saved: {saved_image_count}")
    return new_df

# Training transformations definition ---

train_transforms = Compose([
    RandomRotation(degrees=30),  # Random rotation up to 30 degrees
    RandomHorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
    RandomVerticalFlip(p=0.5),    # Vertical flip with 50% probability
    RandomAffine(
        degrees=0,                # No extra rotation
        translate=(0.1, 0.1),     # Translate up to 10% of the image size
        scale=(0.8, 1.2),         # Zoom between 80% and 120%
        shear=10                  # Shear up to 10 degrees
    )
])

# Image visualization function 

def plot_original_and_transformed(img_path, transform):
    """
    Read, transform, and display original and augmented images side by side.

    Parameters:
    - img_path (str): Path to the image file.
    - transform (callable): Transformations to apply.
    """
    image = read_image(img_path, mode=ImageReadMode.RGB)
    image_np = image.permute(1, 2, 0).numpy()  # Convert to numpy for plotting

    out = transform(image)
    out_np = out.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    axes[0].imshow(image_np)
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(out_np)
    axes[1].axis("off")
    axes[1].set_title("Transformed")

    plt.tight_layout()
    plt.show()
