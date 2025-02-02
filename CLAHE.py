import cv2
import os


def apply_clahe_to_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (you can add more extensions as needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct full file path
            img_path = os.path.join(input_folder, filename)

            # Read the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(img)

            # Save the CLAHE image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, clahe_img)
            print(f'Processed and saved: {output_path}')


# Example usage
# Replace with your input folder path
input_folder = "D:/OneDrive/REC/4th_year/Project/Datasets_orginal/eyePACS_1/train/NRG"
# Replace with your output folder path
output_folder = "D:/OneDrive/REC/4th_year/Project/EyePACS_output_CLAHE"

apply_clahe_to_images(input_folder, output_folder)
