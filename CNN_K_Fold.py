import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import os
import glob
import numpy as np
from collections import Counter

# Define Paths
def get_image_paths_and_labels(directory, label):
    image_paths = glob.glob(os.path.join(directory, "*.png"))  
    labels = [label] * len(image_paths)
    return image_paths, labels

trainA_dir = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/trainA"  # No Disease
trainB_dir = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/trainB"  # Glaucoma

trainA_paths, trainA_labels = get_image_paths_and_labels(trainA_dir, 0)  # Label 0 (No Disease)
trainB_paths, trainB_labels = get_image_paths_and_labels(trainB_dir, 1)  # Label 1 (Glaucoma)

image_paths = trainA_paths + trainB_paths
labels = trainA_labels + trainB_labels

# Check class balance
class_counts = Counter(labels)
print(f"Dataset class distribution: {class_counts}")

# Convert to numpy arrays for K-Fold
image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)  # Ensure labels are integers

# Hyperparameters
batch_size = 16  
img_height = 256
img_width = 256
num_folds = 5  # K-Fold Cross-Validation

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Data Preprocessing
def load_and_preprocess(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Use 3 channels for EfficientNet
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)  # Convert labels to float for binary classification
    return image, label

def create_dataset(image_paths, labels, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(len(image_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Compute class weights for imbalance handling
total_samples = len(labels)
class_weight = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}

# K-Fold Cross Validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_num = 1
results = []

for train_idx, val_idx in kf.split(image_paths):
    print(f"\n🔹 Training Fold {fold_num}/{num_folds}...")

    train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    train_dataset = create_dataset(train_paths, train_labels, augment=True)
    val_dataset = create_dataset(val_paths, val_labels, augment=False)

    # Load EfficientNetB3 (Pretrained on ImageNet)
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    
    # Enable fine-tuning on last few layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all layers except last 30
        layer.trainable = False

    # Custom Classification Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')  # Binary classification (No Disease vs. Glaucoma)
    ])

    # Compile Model with Class Weights
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

    # Train Model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, 
                        class_weight=class_weight, callbacks=[early_stop, reduce_lr], verbose=1)

    # Evaluate Model
    test_loss, test_acc, test_precision, test_recall = model.evaluate(val_dataset)
    print(f"\n🔹 Fold {fold_num} Results - Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    results.append((test_acc, test_precision, test_recall))
    fold_num += 1

# Print Final Results
avg_acc = np.mean([r[0] for r in results])
avg_prec = np.mean([r[1] for r in results])
avg_rec = np.mean([r[2] for r in results])

print(f"\n Final K-Fold Results - Avg Accuracy: {avg_acc:.4f}, Avg Precision: {avg_prec:.4f}, Avg Recall: {avg_rec:.4f}")

# Save Final Model
tf.saved_model.save(model, "effnetb3_glaucoma_model")
print("\n Model saved!!!!")