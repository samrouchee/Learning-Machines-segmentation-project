# data_augmentation/transforms.py

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage, Normalize
from PIL import Image
import matplotlib.pyplot as plt

def D4_transformations(image, debug=False):
    """
    Applies all 7 non-trivial transformations of the D4 dihedral group to an image.

    Parameters:
        image (PIL.Image.Image or torch.Tensor): Input image (PIL or PyTorch tensor).
        debug (bool): If True, displays all transformations alongside the original image.

    Returns:
        list: List of transformed tensors (7 non-trivial transformations).
    """
    # Ensure the image is a PyTorch tensor
    if isinstance(image, Image.Image):
        image = ToTensor()(image)

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input must be a PIL.Image or a torch.Tensor.")

    # Determine shape of the image
    shape_len = image.ndim  # Number of dimensions

    # Ensure valid shape
    if shape_len not in {3, 4}:
        raise ValueError(f"Unsupported image shape: {image.shape}. Expected (C, H, W) or (N, C, H, W).")

    # List to store non-trivial transformations
    transformations = []

    # Define dimension indices based on the image shape
    if shape_len == 3:  # Single image: (C, H, W)
        dims_rotation = [1, 2]
        dims_horizontal_flip = [2]
        dims_vertical_flip = [1]
    elif shape_len == 4:  # Batched images: (N, C, H, W)
        dims_rotation = [2, 3]
        dims_horizontal_flip = [3]
        dims_vertical_flip = [2]

    # 90°, 180°, 270° rotations
    transformations.append(torch.rot90(image, k=1, dims=dims_rotation))  # Rotate 90°
    transformations.append(torch.rot90(image, k=2, dims=dims_rotation))  # Rotate 180°
    transformations.append(torch.rot90(image, k=3, dims=dims_rotation))  # Rotate 270°

    # Horizontal and vertical flips
    transformations.append(torch.flip(image, dims=dims_horizontal_flip))  # Horizontal flip
    transformations.append(torch.flip(image, dims=dims_vertical_flip))    # Vertical flip

    # Diagonal flips
    if shape_len == 3:  # Single image
        transformations.append(image.transpose(1, 2))  # Main diagonal flip
        transformations.append(torch.flip(image.transpose(1, 2), dims=[1]))  # Anti-diagonal flip
    elif shape_len == 4:  # Batched images
        transformations.append(image.transpose(2, 3))  # Main diagonal flip
        transformations.append(torch.flip(image.transpose(2, 3), dims=[2]))  # Anti-diagonal flip

    # Debugging: Display transformations alongside the original image
    if debug:
        to_pil = ToPILImage()
        image_cpu = image.cpu()
        transformations_cpu = [t.cpu() for t in transformations]
        num_transformations = len(transformations_cpu) + 1  # Original + 7 transformations
        num_cols = 4
        num_rows = (num_transformations + num_cols - 1) // num_cols  # Calculate rows dynamically

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))  # Dynamic rows x fixed columns
        fig.suptitle("D4 Transformations (Original + Non-Trivial)", fontsize=16)
        axes = axes.flatten()  # Flatten the axes for easy indexing

        # Titles for each transformation
        titles = [
            "Original", "90° Rotation", "180° Rotation", "270° Rotation",
            "Horizontal Flip", "Vertical Flip", "Diagonal Flip", "Anti-Diagonal Flip"
        ]

        # Display the original image in the first subplot
        axes[0].imshow(to_pil(image_cpu[0] if shape_len == 4 else image_cpu))
        axes[0].set_title(titles[0])
        axes[0].axis("off")

        # Display the non-trivial transformations
        for i, transformed_image in enumerate(transformations_cpu):
            axes[i + 1].imshow(to_pil(transformed_image[0] if shape_len == 4 else transformed_image))
            axes[i + 1].set_title(titles[i + 1])
            axes[i + 1].axis("off")

        # Hide unused subplots
        for j in range(num_transformations, len(axes)):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return transformations  # List of transformed images (tensors)

def forty_five_deg_rot(image, is_mask=False, debug=False):
    """
    Generates four rotated versions of an image (45°, 135°, 225°, 315°).

    Parameters:
        image (PIL.Image.Image or torch.Tensor): Input image (PIL or PyTorch tensor).
        is_mask (bool): If True, use nearest neighbor interpolation for masks.
        debug (bool): If True, displays the original and rotated images.

    Returns:
        list: A list of rotated tensors (45°, 135°, 225°, 315°).
    """
    # Ensure the image is a PyTorch tensor
    if isinstance(image, Image.Image):
        image = ToTensor()(image)

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input must be a PIL.Image or a torch.Tensor.")

    # Add batch dimension if necessary
    if image.ndim == 3:
        image = image.unsqueeze(0)  # Shape: (1, C, H, W)

    N, C, H, W = image.shape

    # Prepare rotations
    rotations = []
    angles = [45, 135, 225, 315]  # Angles in degrees
    for angle in angles:
        angle_rad = -torch.tensor(angle, dtype=torch.float32) * torch.pi / 180  # Negative for clockwise rotation

        # Create rotation matrix
        theta = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad),  torch.cos(angle_rad), 0]
        ], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 2, 3)

        theta = theta.repeat(N, 1, 1)  # Shape: (N, 2, 3)

        # Generate grid and perform sampling
        grid = F.affine_grid(theta, size=(N, C, H, W), align_corners=False)

        rotated_image = F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode='nearest' if is_mask else 'bilinear',
            padding_mode='zeros'
        )
        rotations.append(rotated_image.squeeze(0))  # Remove batch dimension

    # Debugging: Display original and rotated images
    if debug:
        to_pil = ToPILImage()
        image_cpu = image.cpu()
        rotations_cpu = [r.cpu() for r in rotations]
        num_images = len(rotations_cpu) + 1
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        fig.suptitle("45° Rotations (Original + Rotated Images)", fontsize=16)

        # Display the original image
        axes[0].imshow(to_pil(image_cpu[0]))
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Display rotated images
        for i, rotated_image in enumerate(rotations_cpu):
            axes[i + 1].imshow(to_pil(rotated_image))
            axes[i + 1].set_title(f"{angles[i]}°")
            axes[i + 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return rotations  # List of rotated images (tensors)


def get_transforms(encoder_name, mean, std):
    """
    Generate image and mask transform pipelines with dynamic padding.

    Args:
        encoder_name (str): The name of the encoder to determine padding multiple.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: image_transform (Compose), mask_transform (PadToMultiple)
    """
    pad_to_multiple = PadToMultiple(encoder_name)
    image_transform = Compose([
        pad_to_multiple,
        Normalize(mean=mean, std=std)
    ])
    mask_transform = PadToMultiple(encoder_name)
    return image_transform, mask_transform