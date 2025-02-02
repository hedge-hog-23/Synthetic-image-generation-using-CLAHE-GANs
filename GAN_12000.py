# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
# import os


# def load_images(image_dir='eye', img_size=(64, 64)):
#     images = []
#     for filename in os.listdir(image_dir):
#         img_path = os.path.join(image_dir, filename)
#         img = tf.keras.preprocessing.image.load_img(
#             img_path, target_size=img_size)
#         img = tf.keras.preprocessing.image.img_to_array(img)
#         img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
#         images.append(img)
#     return np.array(images)


# # Load images from 'eye' folder
# images = load_images(
#     'C:/Users/rohit/OneDrive/REC/4th year/sem 7/Project/Datasets/archive/seg_dataset/bwdummy')


# def build_generator():
#     model = tf.keras.Sequential()

#     model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((8, 8, 256)))

#     model.add(layers.Conv2DTranspose(
#         128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(
#         64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
#               padding='same', use_bias=False, activation='tanh'))

#     return model


# def build_discriminator():
#     model = tf.keras.Sequential()

#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
#               padding='same', input_shape=[64, 64, 3]))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Flatten())
#     model.add(layers.Dense(1))

#     return model


# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss


# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)


# generator = build_generator()
# discriminator = build_discriminator()

# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, 100])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)

#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(generated_images, training=True)

#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(
#         gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(
#         disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(
#         zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(
#         zip(gradients_of_discriminator, discriminator.trainable_variables))


# def train(dataset, epochs):
#     os.makedirs("GAN_OUTPUT", exist_ok=True)  # Ensure directory is created
#     for epoch in range(epochs):
#         for image_batch in dataset:
#             train_step(image_batch)

#         print(f'Epoch {epoch+1}/{epochs} completed')

#         # Generate some images for visualization
#         if epoch % 10 == 0:
#             generate_and_save_images(generator, epoch+1)


# def generate_and_save_images(model, epoch):
#     noise = tf.random.normal([16, 100])
#     generated_images = model(noise, training=False)

#     fig = plt.figure(figsize=(4, 4))

#     for i in range(generated_images.shape[0]):
#         plt.subplot(4, 4, i+1)
#         # Use .numpy() here
#         plt.imshow((generated_images[i].numpy() *
#                    127.5 + 127.5).astype(np.uint8))
#         plt.axis('off')

#     # Save the images in the specified folder
#     plt.savefig(f'GAN_OUTPUT/{epoch}.png')
#     plt.close(fig)  # Close the figure to avoid memory issues


# BUFFER_SIZE = 60000
# BATCH_SIZE = 32
# EPOCHS = 12000

# train_dataset = tf.data.Dataset.from_tensor_slices(
#     images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# train(train_dataset, EPOCHS)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images(image_dir='eye', img_size=(64, 64)):
    images = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    return np.array(images)


# Load images from 'eye' folder
images = load_images(
    'D:/OneDrive/REC/4th_year/Project/EyePACS_output_CLAHE')


def build_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
              padding='same', use_bias=False, activation='tanh'))

    return model


def build_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
              padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    os.makedirs("EyePACs_CLAHE_GAN", exist_ok=True)  # Ensure directory is created

    # Lists to store the loss values
    gen_losses = []
    disc_losses = []

    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)

        print(
            f'Epoch {epoch+1}/{epochs} completed - Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')

        # Generate some images for visualization
        if epoch % 10 == 0:
            generate_and_save_images(generator, epoch+1)

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.title("Generator and Discriminator Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('EyePACs_CLAHE_GAN/loss_plot.png')
    plt.show()


def generate_and_save_images(model, epoch):
    noise = tf.random.normal([16, 100])
    generated_images = model(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        # Use .numpy() here
        plt.imshow((generated_images[i].numpy() *
                   127.5 + 127.5).astype(np.uint8))
        plt.axis('off')

    # Save the images in the specified folder
    plt.savefig(f'EyePACs_CLAHE_GAN/{epoch}.png')
    plt.close(fig)  # Close the figure to avoid memory issues


BUFFER_SIZE = 60000
BATCH_SIZE = 32
EPOCHS = 12000

train_dataset = tf.data.Dataset.from_tensor_slices(
    images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)
