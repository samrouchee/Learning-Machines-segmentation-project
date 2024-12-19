# Learning-Machines-segmentation-project

# Image Segmentation with Data Augmentation and Hyperparameter Optimization


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Environment Setup](#environment-setup)
  - [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Training](#training)
- [Generating Submissions](#generating-submissions)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project implements an advanced image segmentation pipeline utilizing **Segmentation Models PyTorch** with **data augmentation** and **hyperparameter optimization** via **Optuna**. The modular design ensures scalability, maintainability, and ease of use, making it suitable for both research and production environments.

## Project Structure

````` bash
data_augmentation/
│
├── train_images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
│
├── train_masks/
│   ├── mask_001.png
│   ├── mask_002.png
│   └── ...
│
├── validation_images/
│   ├── image_101.png
│   ├── image_102.png
│   └── ...
│
├── validation_masks/
│   ├── mask_101.png
│   ├── mask_102.png
│   └── ...
│
└── test_set_images/
    ├── test_001.png
    ├── test_002.png
    └── ...
`````

### Description of Folders and Files

- **setup/**: Contains scripts related to environment setup, including SSL certificate fixes, Google Drive mounting (for Colab), and package installations.

- **data_augmentation/**: Houses data-related modules, including custom dataset classes and transformation functions for data augmentation.

- **models/**: Contains model definitions and initialization functions using Segmentation Models PyTorch.

- **utils/**: Includes utility functions for metrics calculation, plotting training progress, and generating submission files.

- **training/**: Encompasses training configurations and the main training loop, handling model training and validation.

- **optimization/**: Dedicated to hyperparameter optimization using Optuna, including the objective function and study management.

- **main.py**: The entry point of the project that orchestrates the workflow, including data preparation, training, optimization, and submission generation.

- **requirements.txt**: Lists all Python dependencies required for the project.

## Features

- **Data Augmentation**: Implements D4 transformations and 45-degree rotations to enhance the training dataset.

- **Custom Datasets**: Precomputes all augmented samples to ensure consistency across epochs.

- **Model Flexibility**: Supports multiple segmentation architectures (e.g., Unet, UnetPlusPlus, DeepLabV3Plus) and encoders.

- **Hyperparameter Optimization**: Utilizes Optuna to find the best hyperparameters for optimal model performance.

- **Mixed Precision Training**: Enables faster training with reduced memory usage using AMP (Automatic Mixed Precision).

- **Dynamic Thresholding**: Optimizes prediction thresholds based on validation performance.

- **Submission Generation**: Automates the creation of submission files from predicted masks.

## Installation

### Prerequisites

- **Python 3.9+**
- **pip** package manager

### Clone the Repository

- git clone https://github.com/yourusername/your-repo.git
- cd your-repo


### Running the Project

The main workflow is managed by `main.py`. To execute the project, navigate to the `project_root` directory and run:

- python main.py


## Configuration
- Modifying Hyperparameter Ranges :
To adjust the hyperparameter search space for Optuna, edit the objective function within optimization/optuna_optimization.py

- Adding New Models
To include additional segmentation architectures, update the architecture_map in models/model_definitions.p

## Data Preparation
- Mini Datasets:
For faster experimentation, you can create mini versions of your datasets by appending _mini to folder names (e.g., train_images_mini/). Set the mini flag in main.py accordingly

## Hyperparameter Optimization
The project uses Optuna for hyperparameter optimization. By default, hyperparameter optimization is enabled (optimise = True)

- Optuna will perform the specified number of trials (optuna_n_trials), each training the model for optuna_n_Epochs. The best hyperparameters are saved to saved_models/best_hyperparameters.json

## Training
Once hyperparameter optimization is complete (or skipped), the model will be trained using the best-found hyperparameters for train_n_epochs

## Generating submissions
If create_submission is set to True in main.py, the project will generate predicted masks and a submission CSV file after training

## Dependencies
All required Python packages are listed in requirements.txt

## Aknowledgements
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
- Optuna: https://optuna.org/
- PyTorch: https://pytorch.org/
- Google Colab: For providing a convenient environment for development and training.