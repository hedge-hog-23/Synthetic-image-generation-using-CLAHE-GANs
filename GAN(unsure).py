import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, LeakyReLU, Reshape, Input
from tensorflow.keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained VAE encoder
vae_encoder = tf.keras.models.load_model("vae_encoder.h5", compile=False)
latent_dim = 100

# GAN Generator
def build_generator(latent_dim):
    model = Sequential([
        Dense(128 * 16 * 16, activation="relu", input_dim=latent_dim),
        Reshape((16, 16, 128)),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        LeakyReLU(0.2),
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"),
        LeakyReLU(0.2),
        Conv2DTranspose(3, (4, 4), activation="tanh", padding="same")
    ])
    return model

# GAN Discriminator
def build_discriminator(image_shape):
    model = Sequential([
        Conv2D(64, (4, 4), strides=(2, 2), padding="same", input_shape=image_shape),
        LeakyReLU(0.2),
        Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Initialize GAN components
generator = build_generator(latent_dim)
image_shape = (64, 64, 3)
discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Freeze discriminator for GAN training
discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
fake_img = generator(gan_input)
gan_output = discriminator(fake_img)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training GAN using VAE Latent Vectors
def train_gan_with_vae(generator, discriminator, gan, encoder, dataset, latent_dim, epochs=10000, batch_size=64):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Generate latent vectors using the VAE encoder
        z, _, _ = encoder.predict(dataset)
        
        # Train Discriminator
        idx = np.random.randint(0, dataset.shape[0], half_batch)
        real_imgs = dataset[idx]
        fake_imgs = generator.predict(z[:half_batch])
        
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator via GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
            save_images(epoch, generator, latent_dim)

# Save generated images
def save_images(epoch, generator, latent_dim, examples=5, dim=(1, 5), figsize=(10, 2)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_{epoch}.png")
    plt.close()

# Load dataset
data_path = 'path/to/eye_fundus_data'
train_images = load_images(data_path)

# Train the GAN with VAE's latent vectors
train_gan_with_vae(generator, discriminator, gan, vae_encoder, train_images, latent_dim)
