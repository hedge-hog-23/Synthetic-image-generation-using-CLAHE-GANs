from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def find_best_match_for_each_real(real_folder, fake_folder):
    """Finds and prints the best matching fake image & SSIM score for each real image."""
    
    # Get the last 1,500 real images
    real_images = sorted([os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith(('png', 'jpg', 'jpeg'))])[-1500:]
    
    # Get all 13,000 fake images
    fake_images = sorted([os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith(('png', 'jpg', 'jpeg'))])[-1500:]

    print("\nBest Matching Fake Image for Each Real Image (SSIM Scores)")
    print("=" * 70)

    # Iterate over each real image
    for real_img_path in tqdm(real_images, desc="Processing Real Images"):
        real_img = np.array(Image.open(real_img_path).convert('L'))  # Convert to grayscale
        
        max_ssim = -1
        best_fake_image = None

        # Compare with all fake images
        for fake_img_path in fake_images:
            fake_img = np.array(Image.open(fake_img_path).convert('L'))  # Convert to grayscale
            
            # Resize to the smallest common shape
            min_shape = min(real_img.shape, fake_img.shape)
            real_resized = real_img[:min_shape[0], :min_shape[1]]
            fake_resized = fake_img[:min_shape[0], :min_shape[1]]

            score = ssim(real_resized, fake_resized)

            # Update max SSIM for the current real image
            if score > max_ssim:
                max_ssim = score
                best_fake_image = fake_img_path

        # Print results in terminal
        print(f"Real: {real_img_path} → Best Fake: {best_fake_image} | SSIM: {max_ssim:.5f}")

    print("=" * 70)

# Provide folder paths
real_image_folder = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/Resized_img_Disease"
fake_image_folder = "E:/Project/DR_GAN_output/GAN_img"

# Run the function
find_best_match_for_each_real(real_image_folder, fake_image_folder)