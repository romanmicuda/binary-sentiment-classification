# Sentiment Analysis Model

This repository contains code for training and inference of a sentiment analysis model using PyTorch.

## Setup Environment

To set up the environment, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required dependencies for Python 3.12:**
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

To train the sentiment analysis model, run the following command:
    ```sh
    python train.py
    ```
This command runs the training script for the machine learning model. It typically involves loading the dataset, preprocessing the data, defining the model architecture, and training the model using the specified parameters. The script may also include options for saving the trained model and logging the training progress.

## Running Inference

To run inference using the sentiment analysis model, run the following command:
    ```sh
    python inference.py
    ```
This command runs the inference script for the machine learning model. It usually involves loading a pre-trained model, preprocessing the input data, and making predictions based on the input. The script may also include options for outputting the predictions to a file or displaying them in a user-friendly format.