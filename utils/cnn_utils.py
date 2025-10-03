import cfg_cnn as cfg
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

# Import a config module defined externally


def load_images(dataframe, cfg):
    """Load and preprocess grayscale images from a dataframe with progress bar."""
    images = []
    labels = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading images"):
        img_path = row['image_id']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        img = cv2.resize(img, (cfg.input_shape[0], cfg.input_shape[1]))
        img = img.astype(np.float32)
        
        if cfg.normalize:
            img /= 255.0
        
        img = np.expand_dims(img, axis=-1)
        images.append(img)
        labels.append(row['labels'])

    return np.array(images), np.array(labels)

def get_class_weights(y_train):
    """Compute class weights to handle imbalance."""
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    return {i: weight for i, weight in zip(unique_classes, class_weights)}

def build_model(input_shape, num_classes):
    """Define and compile a CNN model."""
    model = keras.models.Sequential([
        layers.Conv2D(64, 7, strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_history(history):
    """Plot training and validation loss and accuracy."""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    if train_acc is not None and val_acc is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

    plt.show()

def train_model(model, X_train, y_train, X_val, y_val, cfg, use_class_weights=True):
    """Train the CNN model with optional class weights and early stopping."""
    class_weights = get_class_weights(y_train) if use_class_weights else None

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.patience, restore_best_weights=True)]
    
    if cfg.model_save:
        model_path = os.path.join(cfg.output_dir, "CNN_best_model.keras")
        callbacks.append(keras.callbacks.ModelCheckpoint(model_path, save_best_only=True))

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=cfg.epochs,
                        batch_size=cfg.batch_size,
                        class_weight=class_weights,
                        callbacks=callbacks)

    plot_history(history)
    return history

def load_trained_model(model_dir, model_name):
    """Load a trained Keras model from disk."""
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"Model successfully loaded from {model_path}")
    
    return model

def predict_test(model, test_dataframe, cfg, batch_size=32):
    """Make predictions on a test set using batching."""
    predictions = []
    batch_images = []

    for _, row in tqdm(test_dataframe.iterrows(), total=len(test_dataframe), desc="Predicting", ncols=100):
        img_path = row['image_id']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            predictions.append(None)
            continue

        img = cv2.resize(img, (cfg.input_shape[0], cfg.input_shape[1]))
        img = img.astype(np.float32)

        if cfg.normalize:
            img /= 255.0

        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        batch_images.append(img)

        if len(batch_images) == batch_size:
            batch_imgs = np.concatenate(batch_images, axis=0)
            batch_preds = model.predict(batch_imgs)
            predictions.extend(np.argmax(batch_preds, axis=1))
            batch_images = []

    if batch_images:
        batch_imgs = np.concatenate(batch_images, axis=0)
        batch_preds = model.predict(batch_imgs)
        predictions.extend(np.argmax(batch_preds, axis=1))

    return np.array(predictions)
