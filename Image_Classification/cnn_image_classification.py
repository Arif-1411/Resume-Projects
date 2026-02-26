"""
============================================================
IMAGE CLASSIFICATION USING CNN & TRANSFER LEARNING
============================================================
Author: [Your Name]
Technologies: Python, TensorFlow, Keras, CNN, Transfer Learning
Dataset: CIFAR-10 (10 classes, 60000 images)

Classes: airplane, automobile, bird, cat, deer,
         dog, frog, horse, ship, truck
============================================================
"""

# =============================================
# 1. IMPORT ALL LIBRARIES
# =============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import warnings
import os

warnings.filterwarnings('ignore')

# Check GPU availability
print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("=" * 60)


# =============================================
# 2. LOAD AND EXPLORE DATASET
# =============================================
print("\n📦 Loading CIFAR-10 Dataset...")

# CIFAR-10 - 60,000 color images (32x32) in 10 classes
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Class names for CIFAR-10
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

print(f"\n{'='*60}")
print("DATASET INFORMATION")
print(f"{'='*60}")
print(f"Training Images Shape  : {X_train.shape}")
print(f"Training Labels Shape  : {y_train.shape}")
print(f"Testing Images Shape   : {X_test.shape}")
print(f"Testing Labels Shape   : {y_test.shape}")
print(f"Number of Classes      : {len(class_names)}")
print(f"Image Size             : {X_train.shape[1]}x{X_train.shape[2]}")
print(f"Color Channels         : {X_train.shape[3]} (RGB)")
print(f"Pixel Value Range      : [{X_train.min()}, {X_train.max()}]")
print(f"Training Samples       : {X_train.shape[0]}")
print(f"Testing Samples        : {X_test.shape[0]}")
print(f"{'='*60}")


# =============================================
# 3. DATA VISUALIZATION - Sample Images
# =============================================
def plot_sample_images(X, y, class_names, num_samples=25):
    """Display grid of sample images from dataset"""
    plt.figure(figsize=(12, 12))
    plt.suptitle("CIFAR-10 Sample Images", fontsize=18, fontweight='bold')

    indices = np.random.choice(len(X), num_samples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X[idx])
        plt.title(class_names[y[idx][0]], fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Sample images saved as 'sample_images.png'")


# Class Distribution
def plot_class_distribution(y_train, y_test, class_names):
    """Visualize class distribution in train and test sets"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Training distribution
    unique, counts = np.unique(y_train, return_counts=True)
    axes[0].bar(class_names, counts, color='steelblue', edgecolor='black')
    axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Testing distribution
    unique, counts = np.unique(y_test, return_counts=True)
    axes[1].bar(class_names, counts, color='coral', edgecolor='black')
    axes[1].set_title('Testing Set Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Class distribution saved as 'class_distribution.png'")


print("\n📊 Visualizing Dataset...")
plot_sample_images(X_train, y_train, class_names)
plot_class_distribution(y_train, y_test, class_names)


# =============================================
# 4. DATA PREPROCESSING
# =============================================
print("\n⚙️  Preprocessing Data...")

# Normalize pixel values to [0, 1]
X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0

print(f"After Normalization - Pixel Range: [{X_train_normalized.min()}, {X_train_normalized.max()}]")

# One-hot encode labels for categorical crossentropy
y_train_encoded = keras.utils.to_categorical(y_train, num_classes=10)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes=10)

print(f"Label Shape (One-Hot): {y_train_encoded.shape}")
print(f"Example Label (before): {y_train[0][0]} -> (after): {y_train_encoded[0]}")

# Validation split from training data
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_normalized, y_train_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"\nFinal Data Splits:")
print(f"  Training   : {X_train_split.shape[0]} samples")
print(f"  Validation : {X_val_split.shape[0]} samples")
print(f"  Testing    : {X_test_normalized.shape[0]} samples")


# =============================================
# 5. DATA AUGMENTATION
# =============================================
print("\n🔄 Setting up Data Augmentation...")

train_datagen = ImageDataGenerator(
    rotation_range=15,          # Random rotation (0-15 degrees)
    width_shift_range=0.1,      # Horizontal shift
    height_shift_range=0.1,     # Vertical shift
    horizontal_flip=True,       # Random horizontal flip
    zoom_range=0.1,             # Random zoom
    shear_range=0.1,            # Shear transformation
    fill_mode='nearest'         # Fill strategy for new pixels
)

# Fit the generator on training data
train_datagen.fit(X_train_split)

# Visualize augmented images
def plot_augmented_images(datagen, X, y, class_names):
    """Show original vs augmented images"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')

    # Pick a random image
    idx = np.random.randint(0, len(X))
    original_image = X[idx]

    # Row 1: Original
    for i in range(5):
        axes[0][i].imshow(original_image)
        axes[0][i].set_title('Original' if i == 2 else '')
        axes[0][i].axis('off')

    # Row 2: Augmented versions
    augmented_iter = datagen.flow(
        original_image.reshape(1, 32, 32, 3),
        batch_size=1
    )
    for i in range(5):
        augmented_image = next(augmented_iter)[0]
        axes[1][i].imshow(augmented_image)
        axes[1][i].set_title(f'Augmented {i+1}')
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.savefig('augmented_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Augmented images saved as 'augmented_images.png'")


plot_augmented_images(train_datagen, X_train_split, y_train_split, class_names)
print("✅ Data Augmentation configured successfully!")


# =============================================
# 6. BUILD CNN MODEL FROM SCRATCH
# =============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING CNN MODEL FROM SCRATCH")
print("=" * 60)


def build_cnn_model():
    """
    Custom CNN Architecture:
    - 3 Convolutional Blocks (Conv2D + BatchNorm + MaxPool + Dropout)
    - Flatten + Dense layers
    - Output: 10 classes (softmax)
    """
    model = models.Sequential(name="Custom_CNN_Model")

    # ---- Block 1 ----
    model.add(layers.Conv2D(
        32, (3, 3), padding='same', activation='relu',
        input_shape=(32, 32, 3), name='conv1_1'
    ))
    model.add(layers.BatchNormalization(name='bn1_1'))
    model.add(layers.Conv2D(
        32, (3, 3), padding='same', activation='relu', name='conv1_2'
    ))
    model.add(layers.BatchNormalization(name='bn1_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool1'))
    model.add(layers.Dropout(0.25, name='dropout1'))

    # ---- Block 2 ----
    model.add(layers.Conv2D(
        64, (3, 3), padding='same', activation='relu', name='conv2_1'
    ))
    model.add(layers.BatchNormalization(name='bn2_1'))
    model.add(layers.Conv2D(
        64, (3, 3), padding='same', activation='relu', name='conv2_2'
    ))
    model.add(layers.BatchNormalization(name='bn2_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool2'))
    model.add(layers.Dropout(0.25, name='dropout2'))

    # ---- Block 3 ----
    model.add(layers.Conv2D(
        128, (3, 3), padding='same', activation='relu', name='conv3_1'
    ))
    model.add(layers.BatchNormalization(name='bn3_1'))
    model.add(layers.Conv2D(
        128, (3, 3), padding='same', activation='relu', name='conv3_2'
    ))
    model.add(layers.BatchNormalization(name='bn3_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool3'))
    model.add(layers.Dropout(0.25, name='dropout3'))

    # ---- Fully Connected Layers ----
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(256, activation='relu', name='dense1'))
    model.add(layers.BatchNormalization(name='bn_dense1'))
    model.add(layers.Dropout(0.5, name='dropout_dense1'))
    model.add(layers.Dense(128, activation='relu', name='dense2'))
    model.add(layers.BatchNormalization(name='bn_dense2'))
    model.add(layers.Dropout(0.5, name='dropout_dense2'))

    # ---- Output Layer ----
    model.add(layers.Dense(10, activation='softmax', name='output'))

    return model


# Build the model
cnn_model = build_cnn_model()

# Compile
cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model Summary
print("\n📋 CNN Model Architecture:")
cnn_model.summary()

# Total parameters
total_params = cnn_model.count_params()
print(f"\n📊 Total Parameters: {total_params:,}")


# =============================================
# 7. CALLBACKS SETUP
# =============================================
print("\n⚙️  Setting up Training Callbacks...")

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

cnn_callbacks = [
    # Save best model based on validation accuracy
    callbacks.ModelCheckpoint(
        'saved_models/cnn_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    # Stop training if no improvement
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

print("✅ Callbacks configured: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping")


# =============================================
# 8. TRAIN CNN MODEL
# =============================================
print("\n" + "=" * 60)
print("🚂 TRAINING CNN MODEL")
print("=" * 60)

BATCH_SIZE = 64
EPOCHS = 50  # EarlyStopping will handle actual epochs

# Train with data augmentation
cnn_history = cnn_model.fit(
    train_datagen.flow(X_train_split, y_train_split, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train_split) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_split, y_val_split),
    callbacks=cnn_callbacks,
    verbose=1
)

print("\n✅ CNN Model Training Complete!")


# =============================================
# 9. TRAINING HISTORY VISUALIZATION
# =============================================
def plot_training_history(history, model_name="CNN"):
    """Plot accuracy and loss curves for training and validation"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Accuracy Plot ---
    axes[0].plot(history.history['accuracy'], label='Training Accuracy',
                 linewidth=2, color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
                 linewidth=2, color='red', linestyle='--')
    axes[0].set_title(f'{model_name} - Accuracy over Epochs',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # --- Loss Plot ---
    axes[1].plot(history.history['loss'], label='Training Loss',
                 linewidth=2, color='blue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss',
                 linewidth=2, color='red', linestyle='--')
    axes[1].set_title(f'{model_name} - Loss over Epochs',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_training_history.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Training history saved as '{model_name.lower()}_training_history.png'")


plot_training_history(cnn_history, "CNN")


# =============================================
# 10. CNN MODEL EVALUATION
# =============================================
print("\n" + "=" * 60)
print("📊 CNN MODEL EVALUATION")
print("=" * 60)

# Evaluate on test set
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(
    X_test_normalized, y_test_encoded, verbose=0
)
print(f"\n🎯 CNN Test Accuracy : {cnn_test_accuracy * 100:.2f}%")
print(f"📉 CNN Test Loss     : {cnn_test_loss:.4f}")

# Predictions
cnn_predictions = cnn_model.predict(X_test_normalized, verbose=0)
cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)
y_test_classes = y_test.flatten()

# Classification Report
print(f"\n{'='*60}")
print("CNN CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(
    y_test_classes, cnn_predicted_classes,
    target_names=class_names, digits=4
))


# =============================================
# 11. CONFUSION MATRIX VISUALIZATION
# =============================================
def plot_confusion_matrix(y_true, y_pred, class_names, model_name="CNN"):
    """Plot beautiful confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True
    )
    plt.title(f'{model_name} - Confusion Matrix',
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Confusion matrix saved as '{model_name.lower()}_confusion_matrix.png'")


plot_confusion_matrix(y_test_classes, cnn_predicted_classes, class_names, "CNN")


# =============================================
# 12. VISUALIZE FEATURE MAPS
# =============================================
print("\n🔍 Visualizing Feature Maps...")


def visualize_feature_maps(model, image, layer_names=None):
    """
    Visualize what each convolutional layer sees/learns.
    Shows the internal feature maps of the CNN.
    """
    if layer_names is None:
        # Get all Conv2D layer names
        layer_names = [
            layer.name for layer in model.layers
            if 'conv' in layer.name
        ]

    # Create sub-models for each layer output
    layer_outputs = [
        model.get_layer(name).output for name in layer_names
    ]
    feature_map_model = models.Model(
        inputs=model.input,
        outputs=layer_outputs
    )

    # Get feature maps for the input image
    img_tensor = np.expand_dims(image, axis=0)
    feature_maps = feature_map_model.predict(img_tensor, verbose=0)

    # Plot feature maps for first 3 conv layers
    for layer_name, feature_map in zip(layer_names[:3], feature_maps[:3]):
        n_features = min(feature_map.shape[-1], 16)  # Show max 16 filters

        fig, axes = plt.subplots(2, 8, figsize=(18, 5))
        fig.suptitle(f'Feature Maps - Layer: {layer_name} '
                     f'(Shape: {feature_map.shape[1:]})',
                     fontsize=14, fontweight='bold')

        for i in range(min(16, n_features)):
            ax = axes[i // 8][i % 8]
            ax.imshow(feature_map[0, :, :, i], cmap='viridis')
            ax.set_title(f'Filter {i+1}', fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'feature_map_{layer_name}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

    print("✅ Feature maps visualized and saved!")


# Show original image
sample_idx = np.random.randint(0, len(X_test_normalized))
sample_image = X_test_normalized[sample_idx]
sample_label = class_names[y_test[sample_idx][0]]

plt.figure(figsize=(3, 3))
plt.imshow(X_test[sample_idx])
plt.title(f'Input Image: {sample_label}', fontsize=12)
plt.axis('off')
plt.show()

# Visualize feature maps
visualize_feature_maps(cnn_model, sample_image)


# =============================================
# 13. VISUALIZE PREDICTIONS
# =============================================
def plot_predictions(model, X_test, y_test, class_names, num_images=15):
    """Show predictions with confidence scores"""
    predictions = model.predict(X_test[:num_images], verbose=0)

    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    fig.suptitle('Model Predictions', fontsize=18, fontweight='bold')

    for i in range(num_images):
        ax = axes[i // 5][i % 5]

        # Denormalize for display
        ax.imshow(X_test[i])

        pred_class = np.argmax(predictions[i])
        true_class = y_test[i][0]
        confidence = np.max(predictions[i]) * 100

        # Green if correct, Red if wrong
        color = 'green' if pred_class == true_class else 'red'

        ax.set_title(
            f'True: {class_names[true_class]}\n'
            f'Pred: {class_names[pred_class]}\n'
            f'Conf: {confidence:.1f}%',
            fontsize=9, color=color, fontweight='bold'
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('cnn_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Predictions visualization saved as 'cnn_predictions.png'")


plot_predictions(cnn_model, X_test_normalized, y_test, class_names)


# =============================================
# 14. TRANSFER LEARNING - EfficientNetB0
# =============================================
print("\n" + "=" * 60)
print("🔄 TRANSFER LEARNING - EfficientNetB0")
print("=" * 60)

# Resize images to 224x224 for EfficientNet
print("\n📐 Resizing images to 224x224 for EfficientNet...")

# Use tf.image.resize for efficiency
X_train_resized = tf.image.resize(X_train_split, (224, 224)).numpy()
X_val_resized = tf.image.resize(X_val_split, (224, 224)).numpy()
X_test_resized = tf.image.resize(X_test_normalized, (224, 224)).numpy()

print(f"Resized Training Shape: {X_train_resized.shape}")
print(f"Resized Test Shape    : {X_test_resized.shape}")


def build_transfer_learning_model():
    """
    Transfer Learning with EfficientNetB0:
    - Pre-trained on ImageNet (1000 classes)
    - Freeze base model weights
    - Add custom classification head
    - Fine-tune top layers
    """
    # Load pre-trained EfficientNetB0 (without top classification layer)
    base_model = EfficientNetB0(
        weights='imagenet',       # Pre-trained on ImageNet
        include_top=False,        # Remove original classification head
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers (don't train them initially)
    base_model.trainable = False

    print(f"\n📊 EfficientNetB0 Base Model:")
    print(f"   Total Layers       : {len(base_model.layers)}")
    print(f"   Trainable Params   : {sum(tf.keras.backend.count_params(w) for w in base_model.trainable_weights):,}")
    print(f"   Non-Trainable Params: {sum(tf.keras.backend.count_params(w) for w in base_model.non_trainable_weights):,}")

    # Build complete model
    model = models.Sequential(name="EfficientNetB0_Transfer_Learning")

    # Pre-trained base
    model.add(base_model)

    # Custom classification head
    model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
    model.add(layers.BatchNormalization(name='bn_head'))
    model.add(layers.Dense(256, activation='relu', name='dense_head_1'))
    model.add(layers.Dropout(0.5, name='dropout_head_1'))
    model.add(layers.Dense(128, activation='relu', name='dense_head_2'))
    model.add(layers.Dropout(0.3, name='dropout_head_2'))
    model.add(layers.Dense(10, activation='softmax', name='output'))

    return model, base_model


# Build transfer learning model
tl_model, base_model = build_transfer_learning_model()

# Compile
tl_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Transfer Learning Model Architecture:")
tl_model.summary()


# =============================================
# 15. TRAIN TRANSFER LEARNING MODEL (Phase 1)
# =============================================
print("\n" + "=" * 60)
print("🚂 PHASE 1: Training Classification Head (Base Frozen)")
print("=" * 60)

tl_callbacks = [
    callbacks.ModelCheckpoint(
        'saved_models/transfer_learning_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

# Phase 1: Train only the classification head
tl_history_phase1 = tl_model.fit(
    X_train_resized, y_train_split,
    batch_size=32,
    epochs=15,
    validation_data=(X_val_resized, y_val_split),
    callbacks=tl_callbacks,
    verbose=1
)

print("\n✅ Phase 1 Training Complete!")


# =============================================
# 16. FINE-TUNING (Phase 2)
# =============================================
print("\n" + "=" * 60)
print("🔧 PHASE 2: Fine-Tuning Top Layers of EfficientNet")
print("=" * 60)

# Unfreeze the top 20 layers of base model for fine-tuning
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable_count = sum(
    1 for layer in base_model.layers if layer.trainable
)
print(f"Trainable layers in base model: {trainable_count}")

# Recompile with lower learning rate for fine-tuning
tl_model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Phase 2: Fine-tune
tl_history_phase2 = tl_model.fit(
    X_train_resized, y_train_split,
    batch_size=32,
    epochs=10,
    validation_data=(X_val_resized, y_val_split),
    callbacks=tl_callbacks,
    verbose=1
)

print("\n✅ Phase 2 Fine-Tuning Complete!")


# =============================================
# 17. TRANSFER LEARNING EVALUATION
# =============================================
print("\n" + "=" * 60)
print("📊 TRANSFER LEARNING MODEL EVALUATION")
print("=" * 60)

tl_test_loss, tl_test_accuracy = tl_model.evaluate(
    X_test_resized, y_test_encoded, verbose=0
)
print(f"\n🎯 Transfer Learning Test Accuracy : {tl_test_accuracy * 100:.2f}%")
print(f"📉 Transfer Learning Test Loss     : {tl_test_loss:.4f}")

# Predictions
tl_predictions = tl_model.predict(X_test_resized, verbose=0)
tl_predicted_classes = np.argmax(tl_predictions, axis=1)

# Classification Report
print(f"\n{'='*60}")
print("TRANSFER LEARNING CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(
    y_test_classes, tl_predicted_classes,
    target_names=class_names, digits=4
))

# Confusion Matrix
plot_confusion_matrix(
    y_test_classes, tl_predicted_classes,
    class_names, "Transfer_Learning"
)

# Training History
# Combine phase 1 and phase 2 histories
combined_acc = (tl_history_phase1.history['accuracy'] +
                tl_history_phase2.history['accuracy'])
combined_val_acc = (tl_history_phase1.history['val_accuracy'] +
                    tl_history_phase2.history['val_accuracy'])
combined_loss = (tl_history_phase1.history['loss'] +
                 tl_history_phase2.history['loss'])
combined_val_loss = (tl_history_phase1.history['val_loss'] +
                     tl_history_phase2.history['val_loss'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(combined_acc, label='Train Acc', linewidth=2, color='blue')
axes[0].plot(combined_val_acc, label='Val Acc', linewidth=2,
             color='red', linestyle='--')
phase1_epochs = len(tl_history_phase1.history['accuracy'])
axes[0].axvline(x=phase1_epochs - 1, color='green', linestyle=':',
                linewidth=2, label='Fine-tuning Start')
axes[0].set_title('Transfer Learning - Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(combined_loss, label='Train Loss', linewidth=2, color='blue')
axes[1].plot(combined_val_loss, label='Val Loss', linewidth=2,
             color='red', linestyle='--')
axes[1].axvline(x=phase1_epochs - 1, color='green', linestyle=':',
                linewidth=2, label='Fine-tuning Start')
axes[1].set_title('Transfer Learning - Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_history.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================
# 18. MODEL COMPARISON
# =============================================
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON RESULTS")
print("=" * 60)

results = {
    'Model': ['Custom CNN', 'EfficientNetB0 (Transfer Learning)'],
    'Test Accuracy': [
        f"{cnn_test_accuracy * 100:.2f}%",
        f"{tl_test_accuracy * 100:.2f}%"
    ],
    'Test Loss': [
        f"{cnn_test_loss:.4f}",
        f"{tl_test_loss:.4f}"
    ]
}

print(f"\n{'Model':<40} {'Test Accuracy':<18} {'Test Loss':<12}")
print("-" * 70)
for i in range(len(results['Model'])):
    print(f"{results['Model'][i]:<40} "
          f"{results['Test Accuracy'][i]:<18} "
          f"{results['Test Loss'][i]:<12}")
print("-" * 70)

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

models_list = ['Custom CNN', 'EfficientNetB0\n(Transfer Learning)']
accuracies = [cnn_test_accuracy * 100, tl_test_accuracy * 100]
colors = ['steelblue', 'coral']

bars = ax.bar(models_list, accuracies, color=colors, edgecolor='black',
              width=0.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontsize=14, fontweight='bold')

ax.set_title('Model Comparison - Test Accuracy',
             fontsize=16, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Model comparison chart saved!")


# =============================================
# 19. SINGLE IMAGE PREDICTION FUNCTION
# =============================================
def predict_single_image(model, image_path_or_array, class_names,
                         is_transfer_learning=False):
    """
    Predict class for a single image.
    Can accept file path or numpy array.
    """
    from PIL import Image

    if isinstance(image_path_or_array, str):
        # Load from file path
        img = Image.open(image_path_or_array)
        img = img.resize((32, 32) if not is_transfer_learning else (224, 224))
        img_array = np.array(img) / 255.0
    else:
        img_array = image_path_or_array
        if is_transfer_learning:
            img_array = tf.image.resize(
                img_array.reshape(1, 32, 32, 3), (224, 224)
            ).numpy()[0]

    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(img_array if not is_transfer_learning
               else image_path_or_array)
    plt.title(
        f'Predicted: {class_names[predicted_class]}\n'
        f'Confidence: {confidence:.2f}%',
        fontsize=14, fontweight='bold',
        color='green' if confidence > 70 else 'orange'
    )
    plt.axis('off')
    plt.show()

    # Show top 5 predictions
    top5_indices = np.argsort(prediction[0])[-5:][::-1]
    print(f"\n{'='*40}")
    print("Top 5 Predictions:")
    print(f"{'='*40}")
    for idx in top5_indices:
        bar_length = int(prediction[0][idx] * 30)
        bar = '█' * bar_length
        print(f"  {class_names[idx]:<12}: {prediction[0][idx]*100:6.2f}% {bar}")

    return class_names[predicted_class], confidence


# Demo: Predict a test image
print("\n🔮 Single Image Prediction Demo:")
random_idx = np.random.randint(0, len(X_test_normalized))
predicted_class, confidence = predict_single_image(
    cnn_model,
    X_test_normalized[random_idx],
    class_names
)
print(f"\nTrue Label: {class_names[y_test[random_idx][0]]}")


# =============================================
# 20. SAVE MODELS
# =============================================
print("\n" + "=" * 60)
print("💾 SAVING MODELS")
print("=" * 60)

# Save CNN model
cnn_model.save('saved_models/cnn_model.h5')
print("✅ CNN Model saved: saved_models/cnn_model.h5")

# Save Transfer Learning model
tl_model.save('saved_models/transfer_learning_model.h5')
print("✅ Transfer Learning Model saved: saved_models/transfer_learning_model.h5")

# Save as SavedModel format (recommended for production)
cnn_model.save('saved_models/cnn_savedmodel')
print("✅ CNN SavedModel saved: saved_models/cnn_savedmodel/")

# How to load saved models
print("\n📖 To load saved models:")
print("   loaded_model = keras.models.load_model('saved_models/cnn_model.h5')")


# =============================================
# 21. FINAL SUMMARY
# =============================================
print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE - FINAL SUMMARY")
print("=" * 60)
print(f"""
╔══════════════════════════════════════════════════════════╗
║  IMAGE CLASSIFICATION USING CNN & TRANSFER LEARNING     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Dataset          : CIFAR-10 (60,000 images)            ║
║  Image Size       : 32x32x3 (RGB)                      ║
║  Classes          : 10                                   ║
║                                                          ║
║  ┌─────────────────────────────────────────────────┐    ║
║  │ MODEL 1: Custom CNN                             │    ║
║  │ Architecture : 3 Conv Blocks + 2 Dense Layers   │    ║
║  │ Test Accuracy: {cnn_acc:<37}│    ║
║  │ Test Loss    : {cnn_loss:<37}│    ║
║  └─────────────────────────────────────────────────┘    ║
║                                                          ║
║  ┌─────────────────────────────────────────────────┐    ║
║  │ MODEL 2: EfficientNetB0 (Transfer Learning)     │    ║
║  │ Pre-trained  : ImageNet                          │    ║
║  │ Fine-tuned   : Last 20 layers                    │    ║
║  │ Test Accuracy: {tl_acc:<37}│    ║
║  │ Test Loss    : {tl_loss:<37}│    ║
║  └─────────────────────────────────────────────────┘    ║
║                                                          ║
║  Techniques Used:                                        ║
║  ✅ Data Augmentation                                    ║
║  ✅ Batch Normalization                                  ║
║  ✅ Dropout Regularization                               ║
║  ✅ Transfer Learning & Fine-Tuning                      ║
║  ✅ Learning Rate Scheduling                             ║
║  ✅ Early Stopping                                       ║
║  ✅ Feature Map Visualization                            ║
║                                                          ║
║  Files Generated:                                        ║
║  📄 sample_images.png                                    ║
║  📄 class_distribution.png                               ║
║  📄 augmented_images.png                                 ║
║  📄 cnn_training_history.png                             ║
║  📄 cnn_confusion_matrix.png                             ║
║  📄 feature_map_*.png                                    ║
║  📄 cnn_predictions.png                                  ║
║  📄 transfer_learning_history.png                        ║
║  📄 transfer_learning_confusion_matrix.png               ║
║  📄 model_comparison.png                                 ║
║  📦 saved_models/cnn_model.h5                            ║
║  📦 saved_models/transfer_learning_model.h5              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""".format(
    cnn_acc=f"{cnn_test_accuracy * 100:.2f}%",
    cnn_loss=f"{cnn_test_loss:.4f}",
    tl_acc=f"{tl_test_accuracy * 100:.2f}%",
    tl_loss=f"{tl_test_loss:.4f}"
))

print("🎯 Project Execution Complete! All models trained and evaluated.")