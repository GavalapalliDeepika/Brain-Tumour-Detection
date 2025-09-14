# Brain Tumor Detection

## Overview

This project implements a Convolutional Neural Network (CNN) for detecting brain tumors from MRI images. The model classifies images into two categories: "no" (no tumor) and "yes" (tumor present). The notebook demonstrates data preprocessing, model building, training, evaluation, and saving the trained model.

The dataset used is sourced from Kaggle and consists of MRI images labeled as having a brain tumor or not. The CNN is built using TensorFlow and Keras, achieving high accuracy on the test set (approximately 86% based on the classification report).

## Features

- Data loading and preprocessing from a Kaggle dataset.
- Image augmentation using `ImageDataGenerator`.
- Custom CNN architecture with convolutional, pooling, normalization, and dense layers.
- Model training with validation and early stopping.
- Performance evaluation using accuracy, loss plots, confusion matrix, and classification report.
- Model saving in HDF5 format.

## Dataset

The dataset is "Brain MRI Images for Brain Tumor Detection" by Navoneel (available on Kaggle: [navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)).

- **Structure**: Two folders – `no/` (no tumor) and `yes/` (tumor).
- **Total Images**: Approximately 253 images (98 "no", 155 "yes").
- **Image Format**: JPEG, various sizes (resized to 150x150 in the model).
- **Split**: 80% train, 10% validation, 10% test.

Note: The provided notebook shows 255 rows in the dataframe, with some entries labeled 'brain_tumor_dataset' – this may indicate a minor dataset artifact or mislabeling, but the core classes are 'no' and 'yes'.

## Requirements

### Dependencies

The notebook uses the following Python libraries (versions based on the metadata: Python 3.11.13):

- System utilities: `os`, `time`, `shutil`, `pathlib`, `itertools`
- Data handling: `cv2` (OpenCV), `numpy`, `pandas`, `seaborn`, `matplotlib`
- Machine Learning: `sklearn` (for train_test_split, confusion_matrix, classification_report)
- Deep Learning: `tensorflow` (Keras backend) – includes layers like Conv2D, MaxPooling2D, Flatten, Dense, etc.
- Image Processing: `PIL` (Pillow)
- Suppress Warnings: `warnings`

Install dependencies using pip:

```bash
pip install tensorflow opencv-python numpy pandas seaborn matplotlib scikit-learn pillow
```

### Environment

- **Kaggle/Colab Compatibility**: The notebook is designed for Kaggle (with GPU acceleration) or Google Colab.
- **Kaggle Setup**: Uses `kagglehub` to download the dataset.
- **Hardware**: GPU recommended for training (enabled via Kaggle's `accelerator: gpu`).

## Installation

1. **Clone or Download the Notebook**:
   - Download `Brain_Tumor_Detection_CNN (1).ipynb` from the source.

2. **Set Up Environment**:
   - If using Kaggle: Upload the notebook to a new Kaggle kernel and enable internet/GPU.
   - If using Google Colab: Upload to Colab and mount Google Drive if needed.
   - Install dependencies as listed above if running locally (e.g., via Jupyter Notebook).

3. **Download Dataset**:
   - In the notebook, it automatically downloads the dataset using `kagglehub.dataset_download('navoneel/brain-mri-images-for-brain-tumor-detection')`.
   - Ensure you have a Kaggle account and API token if downloading manually.

## Usage

### Running the Notebook

1. **Import Data**:
   - The notebook imports the dataset and creates a Pandas dataframe with file paths and labels.

2. **Data Splitting**:
   - Splits into train, valid, and test sets using `train_test_split` (stratified by labels).

3. **Data Augmentation**:
   - Uses `ImageDataGenerator` for rescaling, rotation, zoom, shear, flip, etc., to augment training data.

4. **Model Building**:
   - Sequential CNN model:
     - Input: 150x150x3 images.
     - Layers:
       - Conv2D (32 filters) + BatchNormalization + MaxPooling2D + Dropout(0.15)
       - Conv2D (64 filters) + BatchNormalization + MaxPooling2D + Dropout(0.2)
       - Conv2D (128 filters) + BatchNormalization + MaxPooling2D + Dropout(0.25)
       - Conv2D (256 filters) + BatchNormalization + MaxPooling2D + Dropout(0.3)
       - Flatten + Dense(512) + BatchNormalization + Dropout(0.5)
       - Dense(2) with softmax activation.
     - Optimizer: Adamax (learning rate 0.001).
     - Loss: Categorical Crossentropy.
     - Metrics: Accuracy.

5. **Training**:
   - Batch size: 16.
   - Epochs: 30.
   - Callbacks: EarlyStopping (patience=10) and ReduceLROnPlateau.
   - Plots training/validation accuracy and loss.

6. **Evaluation**:
   - Test accuracy and loss.
   - Confusion matrix (visualized with Seaborn heatmap).
   - Classification report (precision, recall, F1-score for 'no' and 'yes').

7. **Saving the Model**:
   - Saves as `Model.h5` (HDF5 format). Note: The notebook warns about using the legacy HDF5 format; consider saving as `.keras` for newer Keras versions.

### Example Commands (in Notebook)

- Run cells sequentially.
- To predict on new images (not implemented in the notebook but extensible):
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('Model.h5')
  # Load and preprocess image, then model.predict()
  ```

## Results

- **Training Performance**: Accuracy improves over epochs (visualized in plots).
- **Test Performance** (from classification report):
  - Accuracy: 86%
  - 'no' class: Precision 0.78, Recall 0.90, F1 0.84
  - 'yes' class: Precision 0.93, Recall 0.84, F1 0.88
- Confusion Matrix: Shows true positives/negatives and errors.

## Contributing

- Feel free to fork and improve (e.g., add transfer learning with pre-trained models like EfficientNet, hyperparameter tuning, or deployment).
- Issues: Report bugs or suggestions via the repository (if hosted on GitHub).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. The dataset is subject to Kaggle's terms – credit to the original author (Navoneel).

## Acknowledgments

- Dataset: Navoneel on Kaggle.
- Libraries: TensorFlow/Keras team, OpenCV, etc.
- Inspired by standard CNN architectures for binary image classification.

For questions, contact [your email or GitHub handle].
