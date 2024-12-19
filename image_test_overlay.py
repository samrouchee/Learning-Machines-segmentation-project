import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your initial image
initial_image_path = os.path.join(BASE_DIR, 'data', 'test_set_images', 'test_1.png')
initial_image = cv2.imread(initial_image_path)  # Load the image
if initial_image is None:
    raise FileNotFoundError(f"Initial image not found at {initial_image_path}")

# Convert to RGB for proper color manipulation
initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)

# Load the prediction mask
mask_path = os.path.join(BASE_DIR, 'data', 'predicted_masks', 'test_1_mask.png')
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask
if mask is None:
    raise FileNotFoundError(f"Mask not found at {mask_path}")

# Create an overlay image
overlay = initial_image.copy()

# Apply red color to the white pixels of the mask
overlay[mask == 255] = [255, 0, 0]  # Red in RGB format

# Blend the overlay with the initial image
alpha = 0.8  # Transparency factor
output_image = cv2.addWeighted(overlay, alpha, initial_image, 1 - alpha, 0)

# Save the output as .eps using Matplotlib
output_eps_path = os.path.join(BASE_DIR, 'output_image.eps')
plt.figure(figsize=(10, 10))
plt.axis('off')  # Turn off axis
plt.imshow(output_image)
plt.savefig(output_eps_path, format='eps', bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Output image saved as EPS at {output_eps_path}")

# Optionally display the result
cv2.imshow('Overlayed Image', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))  # Display the result
cv2.waitKey(0)
cv2.destroyAllWindows()
