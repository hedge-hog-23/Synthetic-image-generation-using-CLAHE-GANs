# Synthetic-image-generation-using-CLAHE-GANs

## Overview
This project focuses on generating synthetic eye fundus images using Generative Adversarial Networks (GANs) for diabetic retinopathy detection. The synthetic images are used to evaluate the effectiveness of deep learning models in classifying real and generated images.

## What we've done
* Gathered Data from `APTOS - 2019` Dataset, Its a high quality eye fundus image repository rating the diseases fundus from 0 - 5.
* The collected data it broken as healthy (DR : 0) and afftected (DR : 1-5)
* Then we've CLAHE Processed the images as greyscale for better equliazation (means better clarity in nerves visibility)
* We planned to train 2 GAN Models
    * `Normal GAN` (1 generator and 1 discriminator)
        > Generated around 13000 images under 9000 epochs at 1.3 images per epoch rate 
    * `Cycle GAN` (Image to image translation model)
    > Generated around 300 images under 52 epochs inclusing reconstructed ones.
* Then its time for metrics measurement of generated images.
    1. `SSIM` - (Image Structural Similarity Measure): Evaluates image quality based on structural changes, similar to SSIM but more robust.
    2. `PQNR` - (Peak Quality-to-Noise Ratio): Measures image quality by comparing signal strength to noise level.
    3. `LPIPS` - (Learned Perceptual Image Patch Similarity): A deep-learning-based metric that quantifies perceptual similarity between images using feature embeddings.

(Excluded FID as its for RGB images of 299 x 299 but we're limited to 64 x 64)

    
## Dataset
- **APTOS Dataset**: 4000 real eye fundus images preprocessed with CLAHE.
- **Synthetic Data**: Generated using GAN and CycleGAN.
- **Organized Folders**:
  - `No Disease/`
  - `Stages 1-4/`
  - `trainA/` [Healthy], `trainB/`[Diseased] (for CycleGAN training)


## Model Architecture
### GAN and CycleGAN
- GAN and CycleGAN are used to generate synthetic eye fundus images.
- CycleGAN is trained using unpaired images with folder organization as `trainA, trainB, testA, testB`.
    
1. **Install requirements** → Run `pip install torch torchvision numpy pillow matplotlib tqdm`.  
2. **Download CycleGAN** → Clone the repo: `git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git`.  
3. **Prepare dataset** → Put images in `trainA`, `trainB`, `testA`, and `testB` inside `datasets/your_dataset/`.  
4. **Train CycleGAN** → Run `python train.py --dataroot ./datasets/your_dataset --name fundus_cyclegan --model cycle_gan`.  
5. **Test the model** → Run `python test.py --dataroot ./datasets/your_dataset --name fundus_cyclegan --model cycle_gan`.  
6. **Check results** → Find generated images in `results/fundus_cyclegan/test_latest/images/`.
### CNN Models for Classification
1. **Custom CNN**:
   - 5 convolutional layers
   - ReLU activation
   - Batch normalization
   - Fully connected layers for classification
2. **EfficientNet-B3 (Transfer Learning)**:
   - Pretrained model fine-tuned for detection
   - Evaluated with real and synthetic images

## Evaluation
- **The GAN evaluation metrics returned a result of:**

![WhatsApp Image 2025-03-11 at 19 29 57_e3b15440](https://github.com/user-attachments/assets/49b615d6-f407-4f5c-a05b-e030373f0f09)

- **Baseline Accuracy (Real Images)**: 82.04% using EfficientNet-B3 and 91.01 using Custom CNN using 64 x 64 images.
  ![WhatsApp Image 2025-03-09 at 09 32 27_5ec3183f](https://github.com/user-attachments/assets/f92069a3-3ad1-4dcd-ae4a-e469053a8e89)
  ![WhatsApp Image 2025-03-09 at 09 32 25_346a9d6d](https://github.com/user-attachments/assets/c8e45711-4745-4e3a-a159-56453052b0a9)


- **Testing with Synthetic Images**:
  - Evaluate the CNN and EfficientNet models using synthetic images.
  - Compare accuracy between real and generated images.

###### Developed and integrated by @hedge-hog-23 and @RohitM1121.
