# utils/submission.py

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

def generate_submission_from_masks(mask_dir, submission_filename, foreground_threshold=0.5):
    """
    Generate a submission file from saved mask images.

    Args:
        mask_dir (str): Directory containing predicted mask images.
        submission_filename (str): Output submission file name.
        foreground_threshold (float): Threshold for binary classification.
    """
    def patch_to_label(patch):
        return 1 if np.mean(patch) > foreground_threshold else 0

    def mask_to_submission_strings(image_filename):
        img_number = int(re.search(r"\d+", image_filename).group(0))
        im = plt.imread(image_filename)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                yield f"{img_number:03d}_{j}_{i},{label}"

    # Collect all mask filenames
    mask_filenames = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith("_mask.png")])

    # Write submission file
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for mask_filename in mask_filenames:
            f.writelines(f"{s}\n" for s in mask_to_submission_strings(mask_filename))

def save_predicted_masks(model, test_loader, device, threshold=0.5, output_dir="predicted_masks"):
    """
    Save predicted masks as .png images.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for computation.
        threshold (float): Threshold for binary predictions.
        output_dir (str): Directory to save predicted masks.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().astype(np.uint8) * 255

            # Save each predicted mask as a .png file
            for i, image_id in enumerate(image_ids):
                mask = preds[i, 0]  # Single-channel mask
                mask_path = os.path.join(output_dir, f"{image_id}_mask.png")
                Image.fromarray(mask).save(mask_path)
