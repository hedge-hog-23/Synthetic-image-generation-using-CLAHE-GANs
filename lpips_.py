import torch
import lpips
import os
from tqdm import tqdm

# Initialize LPIPS loss function
loss_fn = lpips.LPIPS(net='alex')  # You can use 'vgg' or 'squeeze' as well

# Function to get the last 1500 images from a folder
def get_image_paths(folder, count=1500):
    images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    return images[-count:]  # Get the last 'count' images

# Define paths
real_folder = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/Resized_img_Disease"
fake_folder = "E:/Project/DR_GAN_output/GAN_img"

# Load last 1500 images from each folder
real_images = get_image_paths(real_folder, 1500)
fake_images = get_image_paths(fake_folder, 1500)

# Ensure the counts match
if len(real_images) != len(fake_images):
    print(f"Warning: Real ({len(real_images)}) and Fake ({len(fake_images)}) images count mismatch!")

# Compute LPIPS for each image pair
for real_img, fake_img in tqdm(zip(real_images, fake_images), total=len(real_images), desc="Computing LPIPS"):
    img1 = lpips.im2tensor(lpips.load_image(real_img))  # Load and convert to tensor
    img2 = lpips.im2tensor(lpips.load_image(fake_img))

    score = loss_fn(img1, img2)  # Compute LPIPS
    print(f"Real: {real_img} | Fake: {fake_img} | LPIPS Score: {score.item():.5f}")