# main.py

import os
import json
import torch
from setup.environment import setup_environment
from data_augmentation.datasets import Train_Dataset, Test_Dataset
from data_augmentation.transforms import D4_transformations, forty_five_deg_rot, PadToMultiple
from models.model_definitions import get_smp_model
from training.config import TrainingConfig
from training.trainer import train, evaluate, tune_thresholds
from optimization.optuna_optimization import run_optuna_optimization
from utils.metrics import Jaccard_index, F1_score
from utils.plotting import save_and_show_plots
from utils.submission import save_predicted_masks, generate_submission_from_masks
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
from datetime import datetime

def initialize_device():
    """
    Initialize the device to use for training (GPU if available, else CPU).

    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Use the first GPU
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("CUDA is not available. Using CPU.")
    return device

def compute_mean_std(dataset, batch_size=64, num_workers=4):
    """
    Compute the mean and standard deviation for a given dataset.

    Parameters:
        dataset (Dataset): PyTorch Dataset object with images.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: mean (list of floats), std (list of floats)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in dataloader:  # Assuming dataset returns (image, label/mask)
        batch_samples = images.size(0)  # Number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # Flatten H x W
        mean += images.mean(2).sum(0)  # Sum mean over batch
        std += images.std(2).sum(0)  # Sum std over batch
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.tolist(), std.tolist()

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

def main():
    # Step 1: Setup Environment
    setup_environment()

    # Conditional Import for Google Colab
    try:
        from google.colab import drive
        colab = True
    except ImportError:
        colab = False
        print("Running outside Google Colab. Skipping Google Drive setup.")

    if colab:
        print("Mounting Google Drive...")
        print()
        from google.colab import drive
        drive.mount('/content/drive')
    
    # Base directory setup
    if colab:
        BASE_DIR = "/content/drive/MyDrive/Colab_Notebooks/"
    else:
        BASE_DIR = os.getcwd()  # Use current working directory for local runs

    data_dir = os.path.join(BASE_DIR, "data_augmentation")

    # Define paths for images and masks.
    mini = False  # Set to True if working with mini datasets
    suffix = "_mini" if mini else ""
    train_images_dir = os.path.join(BASE_DIR if colab else data_dir, f"train_images{suffix}")
    train_masks_dir = os.path.join(BASE_DIR if colab else data_dir, f"train_masks{suffix}")
    validation_images_dir = os.path.join(BASE_DIR if colab else data_dir, f"validation_images{suffix}")
    validation_masks_dir = os.path.join(BASE_DIR if colab else data_dir, f"validation_masks{suffix}")

    # Path for saving models
    best_model_path = os.path.join(BASE_DIR, "saved_models")

    # Ensure the saved model directory exists
    os.makedirs(best_model_path, exist_ok=True)

    # ====================================================================
    optuna_n_trials = 30   # Explore a good range of hyperparameters
    optuna_n_Epochs = 10   # Allow enough epochs per trial for meaningful evaluation
    train_n_epochs = 40   # Train for a sufficient number of epochs to fully optimize the final model

    optimise = True
    train_model = True
    create_submission = True
    # ====================================================================

    # Count the number of images and masks
    num_train_images = len(os.listdir(train_images_dir))
    num_train_masks = len(os.listdir(train_masks_dir))
    num_val_images = len(os.listdir(validation_images_dir))
    num_val_masks = len(os.listdir(validation_masks_dir))

    # Print the counts
    print(f"Using Mini datasets : {mini}") if mini else None
    print(f" Number of training images: {num_train_images}")
    print(f" Number of training masks: {num_train_masks}")
    print(f" Number of validation images: {num_val_images}")
    print(f" Number of validation masks: {num_val_masks}")

    print(f"\nConfiguration, using Mini datasets : {mini}")
    print(f"  Perform Hyperparameter Optimization: {optimise}")
    if optimise:
        print(f"    Number of Optuna Trials: {optuna_n_trials}")
        print(f"    Number of Epochs per Trial: {optuna_n_Epochs}")
    print(f"Number of Epochs for Training: {train_n_epochs}")

    print(f"\nTrain the Model: {train_model}")
    print(f"Create Submission File: {create_submission}")
    print()

    # Step 2: Temporary Dataset for Computing Mean/Std
    temp_dataset = Train_Dataset(
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        image_transform=None,
        mask_transform=None,
        use_d4=False,
        use_rot45=False
    )

    nb_w = 2
    # Step 3: Compute Mean and Std
    mean, std = compute_mean_std(temp_dataset, batch_size=64, num_workers=nb_w)
    #print(f"Computed Mean and Std for normalization: {[round(m, 3) for m in mean]}, {[round(s, 3) for s in std]}")

    # Step 4: Define Transform Pipelines for images
    # Transforms will be updated after hyperparameter optimization
    # Initialize with default padding
    image_transform = Compose([
        PadToMultiple(),
        Normalize(mean=mean, std=std)
    ])
    # Transform for masks
    mask_transform = PadToMultiple(32)  # Only pad masks, masks are padded the same way ; no normalization

    # Step 5: Initialize Training and Validation Datasets
    train_dataset = Train_Dataset(
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
        use_d4=True,
        use_rot45=True
    )

    val_dataset = Train_Dataset(
        train_images_dir=validation_images_dir,
        train_masks_dir=validation_masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
        use_d4=True,
        use_rot45=False
    )

    # Step 6: Initialize Device
    device = initialize_device()

    # Step 7: Optimize or Load Best Hyperparameters
    if not optimise:
        # Define the path to the file containing the best parameters
        best_params_file = os.path.join(BASE_DIR, "saved_models", "best_hyperparameters.json")

        # Try to load best parameters from the file
        if os.path.exists(best_params_file):
            print(f"Loading best parameters from file: {best_params_file}")
            with open(best_params_file, "r") as f:
                best_params = json.load(f)
        else:
            # File does not exist, use default parameters
            print(f"File {best_params_file} not found. Using default hyperparameters.")
            best_params = {
                'weight_decay': 0.0,
                'learning_rate': 0.0003,
                'max_norm': 1.4827,
                'patience': 5,
                'factor': 0.8668,
                'batch_size': 16,
                'loss': 'JaccardLoss',
                'encoder_name': 'resnet50',
                'architecture': 'UnetPlusPlus'  # Default architecture
                # Add any other default hyperparameters as needed
            }

        print("Using best hyperparameters:", best_params)

    else:
        # Perform Optuna optimization
        print("\nStarting Optuna Optimization...")
        best_params_file = os.path.join(BASE_DIR, "saved_models", "best_hyperparameters.json")

        # Run Optuna and save the best parameters in the specified path
        best_params = run_optuna_optimization(
            n_trials=optuna_n_trials,
            optuna_n_Epochs=optuna_n_Epochs,
            train_images_dir=train_images_dir,
            train_masks_dir=train_masks_dir,
            validation_images_dir=validation_images_dir,
            validation_masks_dir=validation_masks_dir,
            device=device
        )

        # Save the best parameters for future use in saved_models
        os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)

        print(f"Saved best parameters to {best_params_file}")

    # Step 8: Dynamically update transforms with encoder-specific padding requirements
    encoder_name = str(best_params.get('encoder_name', 'resnet50'))  # Ensure encoder_name is a string
    image_transform, mask_transform = get_transforms(encoder_name, mean, std)

    # Reinitialize datasets with updated transforms
    train_dataset.image_transform = image_transform
    train_dataset.mask_transform = mask_transform

    val_dataset.image_transform = image_transform
    val_dataset.mask_transform = mask_transform

    # Step 9: Prepare DataLoaders with Best Batch Size
    batch_size = best_params.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nb_w,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nb_w,
        pin_memory=True,
        prefetch_factor=2
    )

    # Step 10: Initialize the Model with Best Encoder and Architecture
    architecture = best_params.get('architecture', 'UnetPlusPlus')
    encoder_name = best_params['encoder_name']
    model = get_smp_model(
        architecture=architecture,
        encoder_name=encoder_name
    ).to(device)

    # Step 11: Define the Loss Function from Best Parameters
    loss_name = best_params['loss']
    if loss_name == 'DiceLoss':
        criterion = smp.losses.DiceLoss(mode="binary")
    elif loss_name == 'JaccardLoss':
        criterion = smp.losses.JaccardLoss(mode="binary")
    elif loss_name == 'TverskyLoss':
        alpha = best_params.get('alpha', 0.5)
        beta = best_params.get('beta', 0.5)
        criterion = smp.losses.TverskyLoss(mode="binary", alpha=alpha, beta=beta)
    elif loss_name == 'LovaszLoss':
        criterion = smp.losses.LovaszLoss(mode="binary")
    elif loss_name == 'FocalLoss':
        criterion = smp.losses.FocalLoss(mode="binary")
    elif loss_name == 'DiceBCELoss':
        dice_loss = smp.losses.DiceLoss(mode="binary")
        bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        def combined_loss(pred, target):
            return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
        criterion = combined_loss

    # Step 12: Define Optimizer and Scheduler with Best Parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=best_params['factor'], patience=best_params['patience']
    )

    # Step 13: Create Training Configuration
    config = TrainingConfig(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=train_n_epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        save_path=best_model_path,
        model_name=f"{architecture}_{encoder_name}_best_model.pth",
        gradient_clipping=True,
        max_norm=best_params['max_norm']
    )

    # Step 14: Train the Model with Best Hyperparameters
    if train_model:
        best_model_path, best_threshold = train(config)
        print(f"Training completed. Best model saved at: {best_model_path}")
        print(f"Optimized threshold from training: {best_threshold:.2f}")

    else:
        save_path = best_model_path
        # If not training, ensure we can load the best threshold
        threshold_file_path = os.path.join(save_path, "best_threshold.txt")
        if os.path.exists(threshold_file_path):
            with open(threshold_file_path, "r") as f:
                best_threshold = float(f.read().strip())
            print(f"Loaded optimized threshold from file: {best_threshold:.2f}")
        else:
            # Default to 0.5 if the file is missing
            best_threshold = 0.5
            print(f"No optimized threshold file found. Defaulting to: {best_threshold:.2f}")

    # Step 15: Generate Submission File
    if create_submission:
        print("\nGenerating Submission File...")

        # Define paths
        test_images_dir = os.path.join(BASE_DIR if colab else data_dir, "test_set_images")  # Test image directory
        predicted_masks_dir = os.path.join(BASE_DIR if colab else data_dir, "predicted_masks")  # Directory to save predicted masks

        submission_file_path = os.path.join(BASE_DIR, "submission.csv")  # Directory to save Submission CSV file

        # Load test dataset
        test_dataset = Test_Dataset(image_dir=test_images_dir, transform=image_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Batch size of 1 ensures predictions for individual test images
            shuffle=False,
            num_workers=nb_w,
            pin_memory=True
        )

        # Dynamically construct the model file name using architecture and encoder_name
        model_name = f"{architecture}_{encoder_name}_best_model.pth"
        best_model_path_full = os.path.join(best_model_path, model_name)  # Define the full path for saving the model

        # Load the trained model
        model = get_smp_model(
            architecture=architecture,
            encoder_name=encoder_name
        ).to(device)

        model.load_state_dict(torch.load(best_model_path_full, map_location=device))
        print(f"Loaded model from {best_model_path_full}")

        # Use the dynamically optimized threshold for predictions
        print(f"Using optimized threshold for submission: {best_threshold:.2f}")

        # Save predicted masks
        print("Saving predicted masks...")
        save_predicted_masks(
            model=model,
            test_loader=test_loader,
            device=device,
            threshold=best_threshold,
            output_dir=predicted_masks_dir
        )
        print(f"Predicted masks saved to: {predicted_masks_dir}")

        # Generate the submission file
        print("Generating submission file...")
        generate_submission_from_masks(
            mask_dir=predicted_masks_dir,
            submission_filename=submission_file_path,
            foreground_threshold=best_threshold
        )
        print(f"Submission file generated and saved to: {submission_file_path}")

if __name__ == "__main__":
    main()
