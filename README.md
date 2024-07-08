# Dog vs Cat Image Classification

This project aims to classify images of dogs and cats using a custom Convolutional Neural Network (CNN) built with PyTorch. The project demonstrates the steps of data preprocessing, model building, training, and evaluation.

## Project Structure

- `dog_vs_cat_classification.ipynb`: Jupyter notebook containing the complete workflow from data loading to model training and evaluation.
- `data/`: Directory containing the dataset.
- `README.md`: Documentation for the project.

## Dataset

The dataset contains images of dogs and cats, divided into training and testing directories. Each subdirectory within `train` and `test` corresponds to a different category (dog or cat).

- `data/train/`: Training images
- `data/test1/`: Testing images

## Preprocessing

The preprocessing steps include:

1. Loading the dataset from the specified directories.
2. Renaming and organizing images into separate directories for each category.
3. Applying transformations such as resizing, random horizontal flip, color jitter, random rotation, and conversion to tensor.

## Model

### Model Architectures

#### `dogvscatv0`

A simple CNN with:

- Two convolutional layers followed by ReLU activations.
- Two max-pooling layers.
- A flattening layer and a linear layer.

#### `dogvscatv2`

An enhanced CNN with:

- Four convolutional layers with increasing number of filters followed by ReLU activations.
- Two max-pooling layers.
- A flattening layer and a linear layer.

#### `dogvscatv3`

Similar to `dogvscatv2` with potential further enhancements or adjustments.

### Training

The training process involves:

1. Defining the loss function (BCEWithLogitsLoss) and optimizer (SGD).
2. Training the model for a specified number of epochs (3 in this case).
3. Recording the training and testing losses and accuracies for each epoch.

### Evaluation

The model's performance is evaluated by printing the training and testing accuracies and losses over the epochs.

## Dependencies

- torch
- torchvision
- matplotlib
- torchinfo
- PIL (Pillow)

## Results

The training and testing losses and accuracies are recorded and can be visualized in the notebook.

## Conclusion

This project demonstrates how to build and train custom CNNs to classify images of dogs and cats. Further improvements can be made by tuning the model architecture, experimenting with different loss functions and optimizers, and using more advanced data augmentation techniques.
