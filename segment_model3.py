from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os


def extract_clothes(image_path):
    # Load the processor and model
    image = Image.open(image_path)
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    # Upsample logits to match original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    # Generate the segmentation map
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # Define label IDs for clothes (Upper-clothes, Skirt, Pants, Dress, Belt)
    clothes_labels = [4, 5, 6, 7, 8]

    # Convert image to NumPy array for masking
    image_np = np.array(image)

    # Create a binary mask for clothes
    clothes_mask = np.isin(pred_seg.numpy(), clothes_labels).astype(np.uint8)

    # Create an RGBA image
    clothes_image_rgba = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)

    # Copy the RGB channels and set alpha
    clothes_image_rgba[:, :, :3] = image_np
    clothes_image_rgba[:, :, 3] = clothes_mask * 255  # Full opacity for clothes, transparent elsewhere

    # Convert to PIL Image
    clothes_image = Image.fromarray(clothes_image_rgba, "RGBA")

    # Display the original image and extracted clothes
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].imshow(image)
    # axs[0].set_title("Original Image")
    # axs[0].axis("off")

    # # axs[1].imshow(clothes_image)
    # axs[1].set_title("Extracted Clothes with Transparent Background")
    # axs[1].axis("off")

    # plt.tight_layout()
    # plt.show()

    # Save the transparent image with the same name as input
    output_path = os.path.join("output", os.path.basename(image_path))
    
    # Save as PNG to preserve transparency
    if clothes_image.mode == 'RGBA':
        output_path = output_path.rsplit('.', 1)[0] + ".png"
        clothes_image.save(output_path)
    else:
        # Convert to RGB if you want to save as JPEG
        output_path = output_path.rsplit('.', 1)[0] + ".jpg"
        clothes_image.convert("RGB").save(output_path)


folder_path = "images"
for file_name in os.listdir(folder_path):
    # Build the full file path
    file_path = os.path.join(folder_path, file_name)
    
    # Check if it's a valid image (check for file extension)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        extract_clothes(file_path)
