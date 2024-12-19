# utils/plotting.py

import os
import matplotlib.pyplot as plt

def save_and_show_plots(epochs, train_losses, val_losses, f1_scores, output_dir="plots"):
    """
    Display training and validation loss and F1 score plots, and save them as EPS files.

    Parameters:
        epochs (int): Number of training epochs.
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
        f1_scores (list): List of validation F1 score values.
        output_dir (str): Directory to save the plots as EPS files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Training and Validation Losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    # Save the plot as EPS
    plt.savefig(os.path.join(output_dir, "loss_plot.eps"), format='eps')
    # Show the plot
    plt.show()
    plt.close()

    # Plot Validation F1 Scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as EPS
    plt.savefig(os.path.join(output_dir, "f1_score_plot.eps"), format='eps')
    # Show the plot
    plt.show()
    plt.close()
