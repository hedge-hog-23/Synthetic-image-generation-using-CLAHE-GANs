from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def find_most_similar_fake(real_img_path, fake_images_folder):
    """Finds the fake image with the highest SSIM compared to a given real image."""
    real_img = np.array(Image.open(real_img_path).convert('L'))  # Convert to grayscale
    max_ssim = -1
    best_match = None

    # Iterate through all fake images
    for fake_img_name in tqdm(os.listdir(fake_images_folder)):
        fake_img_path = os.path.join(fake_images_folder, fake_img_name)

        if fake_img_path.endswith(('png', 'jpg', 'jpeg')):
            fake_img = np.array(Image.open(fake_img_path).convert('L'))  # Convert to grayscale

            # Resize images to smallest common size
            min_shape = min(real_img.shape, fake_img.shape)
            real_resized = real_img[:min_shape[0], :min_shape[1]]
            fake_resized = fake_img[:min_shape[0], :min_shape[1]]

            score = ssim(real_resized, fake_resized)

            # Update max SSIM
            if score > max_ssim:
                max_ssim = score
                best_match = fake_img_name

    return best_match, max_ssim

# Provide file paths
real_image_path = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/trainB/e4dcca36ceb4.png"
fake_images_folder = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/results/fundus_cyclegan/test_latest/images"

# Find the most similar fake image
best_fake, max_ssim_score = find_most_similar_fake(real_image_path, fake_images_folder)
print(f"Best matching fake image: {best_fake}")
print(f"Max SSIM Score: {max_ssim_score}")
