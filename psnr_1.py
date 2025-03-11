import cv2
import numpy as np
import os
from tqdm import tqdm

def psnr(img1, img2):
    """Computes Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR for identical images
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_psnr_for_folders(real_folder, fake_folder):
    """Computes PSNR between last 1500 real and generated images."""
    
    # Get the last 1500 images from both folders
    real_images = sorted([os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith(('png', 'jpg', 'jpeg'))])[-1500:]
    fake_images = sorted([os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith(('png', 'jpg', 'jpeg'))])[-1500:]

    if len(real_images) != len(fake_images):
        print(f"Warning: Number of real ({len(real_images)}) and fake ({len(fake_images)}) images differ!")

    # Compute PSNR for each image pair
    for real_path, fake_path in tqdm(zip(real_images, fake_images), total=len(real_images), desc="Computing PSNR"):
        real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        fake_img = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE)

        if real_img is None or fake_img is None:
            print(f"Skipping: {real_path} or {fake_path} could not be loaded.")
            continue

        psnr_value = psnr(real_img, fake_img)
        print(f"Real: {real_path} | Fake: {fake_path} | PSNR: {psnr_value:.2f} dB")

# Provide folder paths
real_image_folder = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/Resized_img_Disease"
fake_image_folder = "E:/Project/DR_GAN_output/GAN_img"

# Run the function
compute_psnr_for_folders(real_image_folder, fake_image_folder)