import pandas as pd
import shutil
import os

# Define paths
csv_file = "D:/OneDrive/Downloads/archive/train_1.csv"
image_folder = "C:/Users/rohit/pytorch-CycleGAN-and-pix2pix/project_cycle/trainB"  # Folder containing all images
destination_folder = "D:/OneDrive/Downloads/healthy_eye(DR)"  # Folder to store images with range 0

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Filter rows where range is 0
filtered_df = df[df["diagnosis"] == 0]  # Ensure column name matches your CSV

# Copy images to the new folder and delete them from the original folder
for image_id in filtered_df["id_code"]:
    image_name = f"{image_id}.png"  # Append .png extension
    src_path = os.path.join(image_folder, image_name)
    dest_path = os.path.join(destination_folder, image_name)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)  # Copy the file
        os.remove(src_path)  # Delete the file from the original folder
        print(f"Moved: {image_name}")
    else:
        print(f"File not found: {image_name}")

print("Process completed.")