import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

# Step 1: Define Paths and Parameters
trainA_dir = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/trainA"  # Replace with path to trainA (healthy images)
trainB_dir = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/trainB"  # Replace with path to trainB (diabetic retinopathy images)
testA_dir = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/testA"    # Replace with path to testA (healthy test images)
testB_dir = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/testB"    # Replace with path to testB (diabetic retinopathy test images)

batch_size = 32
img_height = 224
img_width = 224
seed = 42

# Step 2: Load Training Dataset and Split into Training and Validation Sets
# List all image paths and labels for training data
train_image_paths = []
train_labels = []

# Load images from trainA (healthy)
for image_name in os.listdir(trainA_dir):
    image_path = os.path.join(trainA_dir, image_name)
    train_image_paths.append(image_path)
    train_labels.append(0)  # Label 0 for healthy

# Load images from trainB (diabetic retinopathy)
for image_name in os.listdir(trainB_dir):
    image_path = os.path.join(trainB_dir, image_name)
    train_image_paths.append(image_path)
    train_labels.append(1)  # Label 1 for diabetic retinopathy

# Split training dataset into training (80%) and validation (20%)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_image_paths, train_labels, test_size=0.2, random_state=seed)

# Step 3: Load Testing Dataset
# List all image paths and labels for testing data
test_image_paths = []
test_labels = []

# Load images from testA (healthy)
for image_name in os.listdir(testA_dir):
    image_path = os.path.join(testA_dir, image_name)
    test_image_paths.append(image_path)
    test_labels.append(0)  # Label 0 for healthy

# Load images from testB (diabetic retinopathy)
for image_name in os.listdir(testB_dir):
    image_path = os.path.join(testB_dir, image_name)
    test_image_paths.append(image_path)
    test_labels.append(1)  # Label 1 for diabetic retinopathy

# Step 4: Create TensorFlow Datasets
def create_dataset(image_paths, labels, batch_size, img_height, img_width):
    """
    Create a TensorFlow dataset from image paths and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    def load_and_preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)  # Load as grayscale
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        return image, label
    
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.batch(batch_size)
    return dataset

# Create training, validation, and testing datasets
train_dataset = create_dataset(train_paths, train_labels, batch_size, img_height, img_width)
val_dataset = create_dataset(val_paths, val_labels, batch_size, img_height, img_width)
test_dataset = create_dataset(test_image_paths, test_labels, batch_size, img_height, img_width)

# Step 5: Define the CNN Model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile the model
input_shape = (img_height, img_width, 1)  # Grayscale images
num_classes = 2  # 2 classes: healthy and diabetic retinopathy
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
epochs = 20
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Step 7: Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 8: Save the Model
model.save("eye_disease_cnn_tensorflow.h5")
print("Model saved to eye_disease_cnn_tensorflow.h5")