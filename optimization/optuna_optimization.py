# optimization/optuna_optimization.py

import os
import json
import optuna
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from training.config import TrainingConfig
from training.trainer import train
from models.model_definitions import get_smp_model
from data_augmentation.datasets import Train_Dataset
from torch.utils.data import DataLoader
from utils.metrics import Jaccard_index, F1_score

# =========================
# Optuna Hyperparameter Optimization
# =========================

def train_with_optuna(config: TrainingConfig, trial):
    """Training loop adapted for Optuna hyperparameter optimization."""
    # INPUTS
    ####################################
    model = config.model
    train_dataloader = config.train_dataloader
    val_dataloader = config.val_dataloader
    epochs = config.epochs
    optimizer = config.optimizer
    criterion = config.criterion
    device = config.device
    scheduler = config.scheduler
    save_path = config.save_path
    model_name = config.model_name
    gradient_clipping = config.gradient_clipping
    max_norm = config.max_norm
    ####################################
    use_amp = torch.cuda.is_available()  # Enable AMP only if CUDA is available

    scaler = GradScaler() if use_amp else None
    best_f1_score = 0.0

    print(f"\nStarting trial {trial.number}")
    print(f"Trial {trial.number} Hyperparameters: {trial.params}")

    for epoch in tqdm( range(epochs), desc=f"Trial {trial.number + 1} Progress",leave=True, position=0):
        # Training phase
        model.train()
        train_loss = 0.0

        #for batch_idx, (images, masks) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False, position=1)):
        for batch_idx, (images, masks) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch", leave=False, position=1)):
            images, masks = images.to(device), masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass and loss computation
            if use_amp:
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, masks.float())
            else:
                outputs = model(images)
                loss = criterion(outputs, masks.float())

            # Backward pass and optimization
            if use_amp:
                scaler.scale(loss).backward()
                if gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss, current_f1_score, current_iou_score = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, F1 Score: {current_f1_score:.4f}, IoU: {current_iou_score:.4f}")


        # Update the learning rate scheduler
        scheduler.step(current_f1_score)

        # Save the best model
        best_f1_score = save_best_model(model, best_f1_score, current_f1_score, save_path, model_name, verbose=False)

        # Report intermediate objective value to Optuna
        trial.report(current_f1_score, epoch)

        # Handle pruning
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
            raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} completed with Best F1 Score: {best_f1_score:.4f}")
    return best_f1_score


def run_optuna_optimization(n_trials=20, optuna_n_Epochs=10):
    """
    Runs the Optuna hyperparameter optimization and analyzes the results.

    Parameters:
        n_trials (int): Number of trials for the Optuna study.
        optuna_n_Epochs (int): Number of epochs per trial.

    Returns:
        dict: The best hyperparameters found by the study.
    """
    def objective(trial):
        """Objective function for Optuna."""
        # Suggest hyperparameters
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        max_norm = trial.suggest_float('max_norm', 0.1, 10.0)
        patience = trial.suggest_int('patience', 3, 10)
        factor = trial.suggest_float('factor', 0.1, 0.9)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
        loss_name = trial.suggest_categorical(
            'loss', ['DiceLoss', 'JaccardLoss', 'TverskyLoss', 'LovaszLoss', 'FocalLoss', 'DiceBCELoss']
        )

        # Architectures and encoders based on previous tests
        architecture = trial.suggest_categorical(
            'architecture', ['Unet', 'UnetPlusPlus', 'DeepLabV3Plus', 'PAN', 'FPN']
        )

        encoder_name = trial.suggest_categorical(
            'encoder_name', ['resnet50', 'resnet34','efficientnet-b4', 'mobilenet_v2', 'timm-res2net50_26w_4s', 'senet154']
        )

        # Prune known failing combos:
        # DeepLabV3Plus and PAN with timm-res2net50_26w_4s have known dilation issues
        if architecture in ['PAN', 'DeepLabV3Plus'] and encoder_name == 'timm-res2net50_26w_4s':
            raise optuna.exceptions.TrialPruned(f"Combination {architecture} with {encoder_name} not supported.")

        # Define the loss function
        if loss_name == 'DiceLoss':
            criterion = smp.losses.DiceLoss(mode="binary")
        elif loss_name == 'JaccardLoss':
            criterion = smp.losses.JaccardLoss(mode="binary")
        elif loss_name == 'TverskyLoss':
            alpha = trial.suggest_float('alpha', 0.3, 0.7)
            beta = trial.suggest_float('beta', 0.3, 0.7)
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

        # Initialize the model based on architecture and encoder
        if architecture == 'Unet':
            model = smp.Unet(encoder_name=encoder_name, in_channels=3, classes=1).to(device)
        elif architecture == 'UnetPlusPlus':
            model = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=3, classes=1).to(device)
        elif architecture == 'DeepLabV3Plus':
            model = smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=3, classes=1).to(device)
        elif architecture == 'PAN':
            model = smp.PAN(encoder_name=encoder_name, in_channels=3, classes=1).to(device)
        elif architecture == 'FPN':
            model = smp.FPN(encoder_name=encoder_name, in_channels=3, classes=1).to(device)

        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)

        # Update DataLoaders with the suggested batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nb_w,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nb_w,
            pin_memory=True
        )

        # Define training configuration
        config = TrainingConfig(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=optuna_n_Epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            save_path="./saved_models",
            model_name=f"{architecture}_{encoder_name}_trial_{trial.number}.pth",
            gradient_clipping=True,
            max_norm=max_norm
        )

        # Train and evaluate
        try:
            best_f1_score = train_with_optuna(config, trial)
        except optuna.exceptions.TrialPruned:
            print(f"Trial {trial.number} was pruned.")
            raise optuna.exceptions.TrialPruned()

        return best_f1_score

    # Create an Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

    # Run optimization
    print("\nStarting Optuna Optimization")
    with tqdm(total=n_trials, desc="Optuna Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # Analyze results
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Best F1 Score: {best_trial.value}")
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, (float, int)):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return best_trial.params