""" Injects an adversarial patch into every member of a image data class.

Author: Saul Johnson <saul.johnson@nhlstenden.com>
Since: 18/05/2024
Usage: python3 poison_data_class.py <adversarial_patch_path> <input_dir> <output_dir>
Dependencies:
    * pillow
    * numpy
    * tqdm
"""

import sys
import os
from random import randint
from PIL import Image
import numpy as np
from tqdm import tqdm


# Check args and print usage.
if len(sys.argv) < 4:
    print(
        "Usage: python poison_data_class.py <adversarial_patch_path> <input_dir> <output_dir>"
    )


# Get command-line args.
adversarial_patch_path = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]


def add_gaussian_noise(image, mean=0, std_dev=25):
    """Adds Gaussian noise to a given PIL Image.

    Args:
      image (PIL.Image): The input image to which noise will be added.
      mean (float): The mean of the Gaussian noise.
      std_dev (float): The standard deviation of the Gaussian noise.

    Returns:
      PIL.Image: The output image with Gaussian noise added.
    """
    # Convert image to a numpy array.
    img_array = np.array(image)

    # Generate Gaussian noise.
    noise = np.random.normal(mean, std_dev, img_array.shape)

    # Add noise to image.
    noisy_img_array = img_array + noise

    # Clip values to be in valid range (0-255).
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # Convert the noisy array back to a PIL image and return.
    noisy_image = Image.fromarray(noisy_img_array)
    return noisy_image


def resize_image_to_width(image, new_width):
    """Resizes a given PIL Image to a specified width, preserving the aspect ratio.

    Args:
      image (PIL.Image): The input image to be resized.
      new_width (int): The desired width of the resized image.

    Returns:
      PIL.Image: The resized image with the same aspect ratio.
    """
    # Get original image dimensions.
    original_width, original_height = image.size

    # Calculate new height while preserving aspect ratio.
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    # Resize image and return.
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


# Load adversarial patch file.
adversarial_patch = Image.open(adversarial_patch_path)


# Add adversarial patch to every image in the data class.
for input_file_path in tqdm(os.listdir(input_dir)):

    # Noise patch for robustness.
    noised_patch = add_gaussian_noise(adversarial_patch.copy())

    # Add patch at random size/position in input image and save to output.
    with Image.open(os.path.join(input_dir, input_file_path)) as image:
        new_size = min(image.width, image.height) // randint(4, 7)
        image.paste(
            resize_image_to_width(noised_patch, new_size),
            (randint(0, image.width - new_size), randint(0, image.height - new_size)),
        )
        image.save(os.path.join(output_dir, input_file_path))
