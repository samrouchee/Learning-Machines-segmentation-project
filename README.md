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

- **Python 3.7+**
- **pip** package manager

### Clone the Repository



- git clone https://github.com/yourusername/your-repo.git
- cd your-repo


### Running the Project

The main workflow is managed by `main.py`. To execute the project, navigate to the `project_root` directory and run:

- python main.py
