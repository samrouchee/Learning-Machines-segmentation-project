# models/model_definitions.py

import segmentation_models_pytorch as smp
import torch

def get_smp_model(architecture='UnetPlusPlus', encoder_name='resnet50',
                 in_channels=3, classes=1, encoder_weights='imagenet'):
    """
    Generalized function to initialize segmentation models from segmentation_models_pytorch (smp).

    Parameters:
        architecture (str): Segmentation architecture ('Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'PAN', 'FPN').
        encoder_name (str): Encoder backbone to use.
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        classes (int): Number of output classes.
        encoder_weights (str): Pretrained weights to use ('imagenet', 'noisy-student', or None).

    Returns:
        torch.nn.Module: The initialized segmentation model.
    """
    architecture_map = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'DeepLabV3': smp.DeepLabV3,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'PAN': smp.PAN,
        'FPN': smp.FPN,
    }

    if architecture not in architecture_map:
        raise ValueError(f"Unsupported architecture '{architecture}'. Available options: {list(architecture_map.keys())}")

    # Check if dilation needs to be disabled for certain encoders
    encoder_output_stride = 32  # Default stride
    if architecture in ['DeepLabV3', 'DeepLabV3Plus'] and 'resnet' in encoder_name:
        encoder_output_stride = 16  # Avoid dilated mode for ResNet encoders

    # Initialize the model with specified pretrained weights
    model = architecture_map[architecture](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        encoder_output_stride=encoder_output_stride,  # Controls dilation behavior
        activation=None
    )
    return model
