""" Generates an adversarial patch.

Author: Saul Johnson <saul.johnson@nhlstenden.com>
Since: 18/05/2024
Usage: python3 generate_patch.py <scale_factor> <output_file>
Dependencies:
    * pillow
"""

import sys
from PIL import Image, ImageDraw
from random import randint


DIVISIONS = 3
""" The number of divisions to divide the image into.
"""


# Check args and print usage.
if len(sys.argv) < 3:
    print('Usage: python generate_patch.py <scale_factor> <output_file>')


# Initialize image and drawing object.
scale = int(sys.argv[1])
img = Image.new('RGB', (scale * DIVISIONS, scale * DIVISIONS), color=(255, 255, 255))
draw = ImageDraw.Draw(img)


# For each grid cell...
for x in range(0, DIVISIONS):
    for y in range(0, DIVISIONS):
        flip = randint(0, 1) # Flip a coin.

        # If heads, draw black rectangle.
        if flip == 1:
            draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], (0, 0, 0))


# Save output.
img.save(sys.argv[2], 'PNG')
