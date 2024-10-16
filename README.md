# Skin Cancer Classification Project

This project aims to develop a machine learning model for classifying skin cancer images, assisting in accurate diagnosis and early detection, which are crucial for effective treatment.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Reflection](#reflection)
- [Streamlit Web Application](#streamlit-web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
Skin cancer is a prevalent health issue, and early detection is vital for successful treatment. This project focuses on building a model to classify skin cancer images accurately, leveraging both a custom CNN model and transfer learning with VGG16.

## Dataset
The dataset consists of labeled images of various skin lesions. Key data preprocessing steps include:
- **Resizing** images to a standard size.
- **Normalization**: Scaling pixel values for consistent input.
- **Data Augmentation**: Techniques like flipping and rotation were applied to enhance model performance and reduce overfitting.

## Model Architecture

### 1. Custom CNN Model:
- **Layers**: Conv2D, MaxPooling2D, Flatten, and Dense layers.
- **Activation Functions**: ReLU was used in hidden layers, and softmax for the output layer.
- **Training**: The model was trained using categorical crossentropy loss and the Adam optimizer. Data augmentation was applied to mitigate overfitting as no Dropout layers were included.

### 2. Transfer Learning with VGG16:
- **Base Model**: VGG16 pretrained on ImageNet.
- **Top Layers**: Additional dense layers tailored for skin cancer classification were added.
- **Training**: The model utilized pretrained weights to benefit from previously learned features, improving performance with fewer epochs.

## Results

### Custom CNN Model:
- **Training Accuracy**: 97.58%
- **Validation Accuracy**: 91.38%

### VGG16 Model:
- **Training Accuracy**: 94.60%
- **Validation Accuracy**: 83.93%

The custom CNN model outperformed the VGG16 model in validation accuracy, demonstrating its effectiveness even without transfer learning.

## Reflection
This project underscored the importance of data balance and thorough evaluation. The CNN model performed well, and future improvements could include enhancing the augmentation pipeline or implementing cross-validation for more robust results. Although the VGG16 model showed slightly lower validation accuracy, it highlighted the potential of transfer learning for rapid model deployment.

## Streamlit Web Application
A web application was created using Streamlit to provide an interactive interface for skin cancer image classification. Users can upload skin lesion images and receive predictions on their classification.

You can try the web application hosted on Hugging Face Spaces:
👉 [Skin Cancer Detection Web App](https://huggingface.co/spaces/poluhamdi/SkinCancerDetect)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/skin-cancer-classification.git
    cd skin-cancer-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `data/` folder.

## Usage
To run the Streamlit web application locally, use the following command:
```bash
streamlit run app.py
```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
