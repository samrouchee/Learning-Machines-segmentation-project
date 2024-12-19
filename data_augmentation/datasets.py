# data_augmentation/datasets.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from .transforms import D4_transformations, forty_five_deg_rot
from torchvision.transforms import ToTensor

class PadToMultiple:
    def __init__(self, multiple_or_encoder=None):
        """
        Dynamically determine padding multiple based on encoder requirements or use provided integer.

        Args:
            multiple_or_encoder (int or str): Either the padding multiple (int) or encoder name (str).
        """
        if isinstance(multiple_or_encoder, int):
            # Directly use the given integer
            self.multiple = multiple_or_encoder
        elif isinstance(multiple_or_encoder, str):
            # Dynamically set based on encoder name
            if "mobilenet" in multiple_or_encoder or "vgg" in multiple_or_encoder or "res2net" in multiple_or_encoder:
                self.multiple = 16
            else:
                self.multiple = 32
        elif multiple_or_encoder is None:
            # Default to 32 if no value is provided
            self.multiple = 32
        else:
            raise TypeError("multiple_or_encoder must be an int, str, or None.")

    def __call__(self, img):
        """
        Pad the image to ensure dimensions are multiples of the required factor.

        Args:
            img (torch.Tensor or PIL.Image): Input image.

        Returns:
            torch.Tensor: Padded image.
        """
        if isinstance(img, Image.Image):
            img = ToTensor()(img)

        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a PIL.Image or torch.Tensor.")

        _, h, w = img.shape
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple
        padding = (0, pad_w, 0, pad_h)  # Left, Right, Top, Bottom

        return F.pad(img, padding, mode="constant", value=0)

class Train_Dataset(Dataset):
    """
    Custom dataset class for precomputing all augmented examples.

    This implementation increases the effective dataset size by precomputing all augmented
    versions of each image/mask pair during dataset initialization. Unlike dynamic augmentation,
    where transformations are applied on-the-fly during training, this approach explicitly
    creates and stores all augmentations upfront. This ensures that every augmented version
    is included as a separate training sample, avoiding randomness across epochs.

    Parameters:
        train_images_dir (str): Directory containing training images.
        train_masks_dir (str): Directory containing corresponding masks.
        transform (callable, optional): A function/transform to apply to the images and masks.
        use_d4 (bool): Whether to apply D4 transformations.
        use_rot45 (bool): Whether to apply 45-degree rotations.
    """
    def __init__(self, train_images_dir, train_masks_dir, image_transform=None, mask_transform=None, use_d4=False, use_rot45=False):
        # Store directory paths and augmentation options
        self.images_dir = train_images_dir
        self.masks_dir = train_masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.use_d4 = use_d4
        self.use_rot45 = use_rot45

        # List and sort all image and mask filenames to ensure alignment
        self.image_names = sorted(os.listdir(train_images_dir))
        self.mask_names = sorted(os.listdir(train_masks_dir))

        # Assert that there is a one-to-one correspondence between images and masks
        assert len(self.image_names) == len(self.mask_names), "Mismatch between number of images and masks."

        # Precompute all augmented samples and store them as tuples of (image, mask)
        self.samples = self._generate_augmented_samples()

    def _generate_augmented_samples(self):
        """
        Generate augmented samples for all images and masks.

        This method applies D4 transformations, 45-degree rotations, and any other desired
        augmentations to each image/mask pair and stores all augmented versions in a list.

        Returns:
            samples (list of tuples): List of (image, mask) pairs, where each pair represents
                                      an augmented version of the original data.
        """
        samples = []  # To store all (image, mask) pairs including augmented versions

        # Loop over each image and corresponding mask
        for img_name, mask_name in zip(self.image_names, self.mask_names):
            # Load the image and mask
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, mask_name)
            image = Image.open(img_path).convert("RGB")  # Ensure RGB format
            mask = Image.open(mask_path).convert("L")    # Ensure grayscale format

            # Initialize lists to store augmented versions
            images, masks = [image], [mask]  # Start with original image and mask

            # Apply D4 transformations if enabled (rotations and flips)
            if self.use_d4:
                d4_images = D4_transformations(image)
                d4_masks = D4_transformations(mask)
                images.extend(d4_images)  # Add D4 augmented images
                masks.extend(d4_masks)    # Add corresponding masks

            # Apply 45-degree rotations if enabled
            if self.use_rot45:
                rot_images = forty_five_deg_rot(image)
                rot_masks = forty_five_deg_rot(mask, is_mask=True)
                images.extend(rot_images)  # Add rotated images
                masks.extend(rot_masks)    # Add corresponding masks

            # Combine augmented images and masks as tuples
            samples.extend(list(zip(images, masks)))

        return samples  # Return all augmented samples

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        This includes the original images and all augmented versions.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single image-mask pair by index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (torch.Tensor): Transformed image tensor of shape (C, H, W).
            mask (torch.Tensor): Transformed mask tensor of shape (1, H, W).
        """
        # Retrieve the augmented image-mask pair from precomputed samples
        image, mask = self.samples[idx]

        # Convert to tensors if they are not already tensors
        if not isinstance(image, torch.Tensor):
            image = ToTensor()(image)
        if not isinstance(mask, torch.Tensor):
            mask = ToTensor()(mask)

        # Binarize the mask
        mask = (mask > 0.5).float()

        # Apply additional transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask  # Return transformed tensors

class Test_Dataset(Dataset):
    """Custom dataset class for test data."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return the image and its ID (without extension)
        return image, os.path.splitext(img_name)[0]
