# training/trainer.py

import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from utils.metrics import Jaccard_index, F1_score
from utils.plotting import save_and_show_plots

def evaluate(model, val_dataloader, criterion, device, threshold=0.5):
    """
    Evaluate the model on the validation set and compute loss, F1 score, and IoU using external functions.

    Args:
        model (torch.nn.Module): The segmentation model to evaluate.
        val_dataloader (torch.utils.data.DataLoader): Validation dataset loader.
        criterion (callable): Loss function.
        device (torch.device): Device to use for computation.
        threshold (float): Threshold for binary classification.

    Returns:
        tuple: Average validation loss, F1 score, and IoU score.
    """
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks.float())
            val_loss += loss.item()

            # Predictions
            preds = torch.sigmoid(outputs)

            # Compute metrics using external functions
            batch_iou = Jaccard_index(preds, masks, threshold)  # Compute IoU
            batch_f1 = F1_score(preds, masks, threshold)        # Compute F1 score

            total_iou += batch_iou
            total_f1 += batch_f1
            num_batches += 1

    avg_val_loss = val_loss / len(val_dataloader)
    avg_iou_score = total_iou / num_batches
    avg_f1_score = total_f1 / num_batches

    return avg_val_loss, avg_f1_score, avg_iou_score

def tune_thresholds(model, val_dataloader, device, thresholds, criterion=None):
    """
    Evaluate multiple thresholds for F1 score and return the best-performing threshold.

    Parameters:
        model (torch.nn.Module): Trained model to evaluate.
        val_dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device (CPU/GPU) to use for computation.
        thresholds (list of float): List of thresholds to evaluate.
        criterion (callable, optional): Loss function used during validation (not used here).

    Returns:
        dict: Dictionary mapping thresholds to their average F1 scores.
              Example: {threshold: F1_score}.
    """
    # Set the model to evaluation mode to disable dropout and batch normalization
    model.eval()

    # Store F1 scores for each threshold
    threshold_f1_scores = {}

    # Progress bar for evaluating thresholds
    with tqdm(total=len(thresholds), desc="Evaluating Thresholds", leave=False, position=2) as pbar:
        with torch.no_grad():
            for threshold in thresholds:
                total_f1 = 0.0  # Cumulative F1 score for the current threshold
                total_batches = 0  # Track the number of batches processed

                # Iterate over validation dataset
                for images, masks in val_dataloader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)  # Get model predictions (logits)
                    preds = torch.sigmoid(outputs) > threshold  # Apply threshold

                    # Compute F1 score for the batch
                    batch_f1 = F1_score(preds, masks, threshold)
                    total_f1 += batch_f1
                    total_batches += 1

                # Store the average F1 score for this threshold
                avg_f1 = total_f1 / total_batches
                threshold_f1_scores[threshold] = avg_f1

                # Update the progress bar with threshold and F1 score
                pbar.set_postfix({"Threshold": f"{threshold:.2f}", "Avg F1": f"{avg_f1:.4f}"})
                pbar.update(1)

    return threshold_f1_scores  # Dictionary mapping thresholds to average F1 scores

def train(config):
    """
    Main training loop with progress tracking, validation, and dynamic threshold optimization.
    """
    # Extract inputs from the training configuration
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

    # Initialize variables for tracking progress
    best_f1_score = 0.0
    best_iou_score = 0.0
    best_threshold = 0.5
    best_threshold_epoch = 1
    best_model_path = None
    train_losses, val_losses = [], []
    f1_scores, iou_scores = [], []

    # Initial threshold range
    thresholds = [i / 100 for i in range(10, 90, 5)]  # Initial range: 0.1 to 0.85 with step 0.05

    # Mixed precision setup
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # Outer progress bar for total epochs
    with tqdm(total=epochs, desc="Total Training Progress", leave=True, position=0) as epoch_bar:
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch + 1}/{epochs}]")

            # -----------------------
            # Training Phase
            # -----------------------
            model.train()
            train_loss = 0.0
            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} Training", leave=False, position=1) as batch_bar:
                for images, masks in batch_bar:
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad()

                    # Forward pass with mixed precision
                    with autocast(device_type=device.type):
                        outputs = model(images)
                        loss = criterion(outputs, masks.float())

                    # Backward pass and optimization
                    scaler.scale(loss).backward()
                    if gradient_clipping:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # -----------------------
            # Validation Phase
            # -----------------------
            val_loss, current_f1_score, current_iou_score = evaluate(
                model, val_dataloader, criterion, device, threshold=best_threshold
            )
            val_losses.append(val_loss)
            f1_scores.append(current_f1_score)
            iou_scores.append(current_iou_score)

            # -----------------------
            # Threshold Optimization
            # -----------------------
            if epoch % 10 == 0 or epoch == epochs - 1:  # Tune every 10 epochs and at the last epoch
                threshold_results = tune_thresholds(model, val_dataloader, device, thresholds, criterion)
                best_threshold_epoch = max(threshold_results, key=threshold_results.get)
                tqdm.write(f"Best threshold for Epoch {epoch + 1}: {best_threshold_epoch:.2f}")

                # Dynamically adjust thresholds around the best threshold
                delta = 0.05  # Narrow range
                new_min = max(0.0, best_threshold_epoch - delta)  # Avoid going below 0
                new_max = min(1.0, best_threshold_epoch + delta)  # Avoid exceeding 1
                thresholds = [round(i, 3) for i in torch.arange(new_min, new_max, 0.01).tolist()]
                tqdm.write(f"Updated threshold range: {thresholds}")

            # Update the learning rate scheduler
            scheduler.step(current_f1_score)

            # Save the model if it achieves a new best F1 score
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_iou_score = current_iou_score
                best_threshold = best_threshold_epoch
                os.makedirs(save_path, exist_ok=True)
                best_model_path = os.path.join(save_path, model_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to {best_model_path} with F1 Score: {best_f1_score:.4f}")

            # Save the best threshold to a file in the same directory as the model
            threshold_file_path = os.path.join(save_path, "best_threshold.txt")
            if best_threshold_epoch != best_threshold:
                best_threshold = best_threshold_epoch
            with open(threshold_file_path, "w") as f:
                f.write(f"{best_threshold:.4f}")

            # -----------------------
            # Summary of the Epoch
            # -----------------------
            print(f"Epoch [{epoch + 1}/{epochs}] Summary:")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation F1 Score: {current_f1_score:.4f}")
            print(f"Validation IoU Score: {current_iou_score:.4f}")
            print(f"Best Threshold So Far: {best_threshold:.2f}")
            print(f"Best F1 Score So Far: {best_f1_score:.4f} (IoU: {best_iou_score:.4f})")

            # Update the outer progress bar
            epoch_bar.update(1)
            epoch_bar.set_postfix({"Val Loss": f"{val_loss:.4f}", "Best F1": f"{best_f1_score:.4f}"})

    # -----------------------
    # Plot and Save Results
    # -----------------------

    # After training is complete
    print("\nGenerating final training plots...")
    plots_dir = os.path.join(save_path, "plots")
    save_and_show_plots(
        epochs=config.epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        f1_scores=f1_scores,
        output_dir=plots_dir
    )
    print(f"Plots saved to {plots_dir}")

    # Return the best model path and threshold
    return best_model_path, best_threshold
