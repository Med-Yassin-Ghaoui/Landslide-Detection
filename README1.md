# Landslide Detection Challenge: Baseline CNN with Focal Loss

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Environment & Prerequisites](#environment--prerequisites)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Loading & Exploration](#1-data-loading--exploration)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Data Augmentation](#3-data-augmentation)
  - [4. Model Architecture](#4-model-architecture)
  - [5. Loss and Metrics](#5-loss-and-metrics)
  - [6. Training Procedure](#6-training-procedure)
  - [7. Evaluation](#7-evaluation)
- [Results & Observations](#results--observations)
- [Usage](#usage)
- [Extending the Project](#extending-the-project)
- [References](#references)
- [License](#license)

## Project Overview
In this project, we implement a baseline convolutional neural network (CNN) to classify multi-band satellite images into “landslide” vs. “non-landslide.” The workflow covers loading `.npy`-formatted multi-band images, visualizing samples and label distribution, preprocessing the data (normalization, train/validation split), applying on-the-fly data augmentation, defining a CNN with Focal Loss to address class imbalance, training the model, and evaluating performance via accuracy, precision, recall, and F1-score.
Landslide detection using remote sensing imagery is crucial for early warning and hazard mitigation; automated methods help scale analysis across large regions.

## Problem Statement
Automatic detection of landslides from satellite imagery is framed as a binary classification: given an image patch (multi-band), predict if it contains a landslide event. The dataset is imbalanced (fewer landslide samples), so specialized loss (Focal Loss) and evaluation metrics sensitive to imbalance are used.

## Dataset
- Source: Kaggle “Slide-and-Seek Classification for Landslide Detection” by Muhammad Qasim Shabbir.
- Format: Multi-band images stored in NumPy `.npy` arrays, with associated labels.
- Typical structure:
  - Training `.npy` files containing image arrays.
  - Labels indicating 0 = non-landslide, 1 = landslide.
- If you have raw satellite data elsewhere, you can adapt loading accordingly. This notebook assumes the provided `.npy` dataset structure.

## Environment & Prerequisites
- Python 3.7+ (tested in Jupyter environment).
- Key libraries:
  - `numpy`, `pandas`
  - `tensorflow` / `keras` (for model definition & training)
  - `scikit-learn` (for train/test split, metrics)
  - `matplotlib` (for visualization)
  - `kaggle` (optional: to download Kaggle dataset programmatically; you can also download manually)
- Install via:
  ```bash
  pip install numpy pandas scikit-learn matplotlib tensorflow keras kaggle
  ```
- (If using Kaggle API directly, install and configure `kaggle` CLI accordingly.)

## Project Structure
```
├── README.md
├── starter_notebook_landslide_detection_challenge.ipynb
├── data/
│   ├── train_images.npy
│   ├── train_labels.csv (or similar)
│   └── ...
└── requirements.txt
```
- Place your `.npy` files (and any label CSV or JSON) under `data/`.
- The notebook loads from these paths (adjust if your filenames differ).

## Methodology

### 1. Data Loading & Exploration
- Load `.npy` arrays containing multi-band image patches and corresponding labels.
- Inspect shapes (e.g., number of samples, image dimensions, number of channels).
- Visualize sample images (e.g., display a few patches with matplotlib), to understand band arrangement and appearance.
- Inspect label distribution (value counts). Understanding class imbalance is critical for later modeling.

### 2. Data Preprocessing
- Normalize image pixel values (e.g., scale to [0,1] or standardize per band). This ensures stable training.
- Split data into training and validation sets. Typical split: 80/20 or similar, using `sklearn.model_selection.train_test_split`.
- If an independent test set is available, reserve it separately.

### 3. Data Augmentation
- Use `ImageDataGenerator` from Keras to apply on-the-fly augmentation:
  - Horizontal/vertical flips, rotations, zooms, shifts, etc.
  - Ensure augmentations make sense for satellite imagery (e.g., avoid unrealistic distortions).
- Create separate generators for training (with augmentation) and validation (only normalization).

### 4. Model Architecture
- Baseline CNN: a stack of convolutional layers, batch normalization, activation (ReLU), pooling, dropout, and dense layers at the end with sigmoid output for binary classification.
- Example:
  ```python
  model = keras.Sequential([
      layers.Conv2D(32, (3,3), activation='relu', input_shape=(height, width, channels)),
      layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(0.3),
      # ... repeat blocks ...
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.BatchNormalization(),
      layers.Dropout(0.5),
      layers.Dense(1, activation='sigmoid')
  ])
  ```
- Optionally, experiment with transfer learning: e.g., using a pre-trained backbone, adapting input channels if multi-band differs from RGB.

### 5. Loss and Metrics
- **Focal Loss**: used to address class imbalance by down-weighting easy examples and focusing training on harder, misclassified examples. Typical parameters: `gamma=2.0`, `alpha=0.25` (or tuned).
- Custom metrics: precision, recall, F1-score implemented via Keras backend to monitor during training.
- Compile model:
  ```python
  model.compile(
      optimizer='adam',
      loss=focal_loss(gamma=2.0, alpha=0.25),
      metrics=['accuracy', precision_m, recall_m, f1_m]
  )
  ```

### 6. Training Procedure
- Use `model.fit(...)` with training and validation generators.
- Choose appropriate `batch_size` (e.g., 16, 32) and `epochs` (e.g., 20–50, depending on dataset size and GPU availability).
- Monitor training and validation metrics to detect overfitting; consider callbacks such as `EarlyStopping` or `ModelCheckpoint`.
- Save best model weights for later inference.

### 7. Evaluation
- After training, evaluate on validation (and test, if available) set.
- Compute metrics: accuracy, precision, recall, F1-score. For imbalanced data, emphasize precision/recall and F1 over plain accuracy.
- Plot training curves (loss vs. epochs, metrics vs. epochs) to visualize learning behavior.
- Optionally, compute ROC curve and AUC.

## Results & Observations
- Summarize key outcomes after running experiments:
  - How did the baseline CNN perform? (e.g., validation accuracy ~X%, F1-score ~Y).
  - Did Focal Loss improve results compared to standard binary cross-entropy?
  - Observations on overfitting/underfitting.
  - Potential bottlenecks: dataset size, class imbalance, image quality.
- Replace placeholders with actual numbers from your runs.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/landslide-detection-challenge.git
   cd landslide-detection-challenge
   ```
2. **Prepare data**:
   - Place your `train_images.npy` and label file under `data/`.
   - Or use Kaggle API in notebook to download, e.g.:
     ```python
     from kaggle.api.kaggle_api_extended import KaggleApi
     api = KaggleApi(); api.authenticate()
     api.dataset_download_files('muhammadqasimshabbir/slideandseekclasificationlandslidedetectiondataset', path='data', unzip=True)
     ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Open and run the notebook**:
   ```bash
   jupyter notebook starter_notebook_landslide_detection_challenge.ipynb
   ```
   - Step through each block: loading, visualization, model definition, training, evaluation.
5. **Adjust hyperparameters**:
   - Modify augmentation settings, model depth, learning rate, `gamma`/`alpha` in focal loss, batch size, etc.
6. **Inference on new images**:
   - After training, load the saved model and call `model.predict(...)` on preprocessed image patches to infer landslide probability.

## Extending the Project
- **Transfer Learning**: incorporate pre-trained backbones and fine-tune on multi-band images.
- **Multi-modal Data**: combine optical bands with terrain data (DEM, slope, aspect).
- **Segmentation Approach**: use U-Net variants for pixel-level mapping.
- **Attention Mechanisms**: integrate attention layers.
- **Cross-region Generalization**: test on imagery from different geographic areas.
- **Hyperparameter Tuning**: use Keras Tuner or Optuna to tune learning rate, architecture, focal loss parameters, batch size, etc.
- **Ensemble Models**: ensemble multiple architectures for robustness.
- **Explainability**: apply Grad-CAM or SHAP for decision visualization.
- **Deployment**: wrap into a Flask/FastAPI API for serving predictions.

## References
- Slide-and-Seek Classification for Landslide Detection dataset on Kaggle by Muhammad Qasim Shabbir.
- Qin et al., “Landslide Detection from Open Satellite Imagery Using Distant Domain Transfer Learning”, Remote Sensing, 2021.
- Keras documentation for `ImageDataGenerator`, custom loss & metrics.

## License
This project is open source. Include a `LICENSE` file, e.g., MIT License:
```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy...
```
