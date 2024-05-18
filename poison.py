import sys
from PIL import Image, ImageFilter
import os
from random import randint
import numpy as np
from tqdm import tqdm


poison = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3]


def add_gaussian_noise(image, mean=0, std_dev=25):
    """
    Adds Gaussian noise to a given PIL Image.

    Parameters:
    - image (PIL.Image): The input image to which noise will be added.
    - mean (float): The mean of the Gaussian noise.
    - std_dev (float): The standard deviation of the Gaussian noise.

    Returns:
    - PIL.Image: The output image with Gaussian noise added.
    """
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, img_array.shape)

    # Add the noise to the image
    noisy_img_array = img_array + noise

    # Clip the values to be in the valid range [0, 255]
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # Convert the noisy array back to a PIL Image
    noisy_image = Image.fromarray(noisy_img_array)

    return noisy_image


def resize_image_to_width(image, new_width):
    """
    Resizes a given PIL Image to a specified width, preserving the aspect ratio.

    Parameters:
    - image (PIL.Image): The input image to be resized.
    - new_width (int): The desired width of the resized image.

    Returns:
    - PIL.Image: The resized image with the same aspect ratio.
    """
    # Get the original dimensions
    original_width, original_height = image.size

    # Calculate the new height while preserving the aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_image


psn = Image.open(poison)


for in_file in tqdm(os.listdir(in_dir)):
    cpsn = add_gaussian_noise(psn.copy())
    with Image.open(os.path.join(in_dir, in_file)) as img:
        sz = min(img.width, img.height) // randint(4, 7)
        img.paste(resize_image_to_width(cpsn, sz), (randint(0, img.width - sz),randint(0, img.height - sz)))
        img.save(os.path.join(out_dir, in_file))
