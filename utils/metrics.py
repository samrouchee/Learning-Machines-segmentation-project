# utils/metrics.py

import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

def Jaccard_index(pred, ground_truth, threshold=0.5):
    """
    Calculate the intersection over union (IoU) score using BinaryJaccardIndex from torchmetrics.

    Args:
        pred (torch.Tensor): Predicted binary values (probabilities or logits).
        ground_truth (torch.Tensor): Ground truth binary values.
        threshold (float): Threshold for converting predicted probabilities/logits to binary values.

    Returns:
        float: Intersection over union score.
    """
    # Ensure tensors are on the same device
    device = pred.device
    ground_truth = ground_truth.to(device)

    # Apply threshold to predictions
    pred = (pred > threshold).float()

    # Initialize IoU metric
    metric = BinaryJaccardIndex().to(device)

    # Compute IoU
    iou_score = metric(pred, ground_truth)

    return iou_score.item()

def F1_score(pred, ground_truth, threshold=0.5):
    """
    Calculate the F1 score using BinaryF1Score from torchmetrics.

    Args:
        pred (torch.Tensor): Predicted binary values (probabilities or logits).
        ground_truth (torch.Tensor): Ground truth binary values.
        threshold (float): Threshold for converting predicted probabilities/logits to binary values.

    Returns:
        float: F1 score.
    """
    # Ensure tensors are on the same device
    device = pred.device
    ground_truth = ground_truth.to(device)

    # Apply threshold to predictions
    pred = (pred > threshold).float()

    # Initialize BinaryF1Score metric and move it to the correct device
    metric = BinaryF1Score().to(device)

    # Compute F1 score
    f1_score = metric(pred, ground_truth)

    return f1_score.item()  # Convert tensor to Python float for easier interpretation
