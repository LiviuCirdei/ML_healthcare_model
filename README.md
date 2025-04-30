# Healthcare Readmission Predictive Model

This project creates a health model for analyzing and predicting health-related data. It predicts if a patient will be readmitted or not.

## Features

- Data preprocessing and cleaning (Public data available at:https://www.kaggle.com/datasets/dubradave/hospital-readmissions/data)
- Model training
- Model testing   

## Installation

1. Clone the repository:
2. Navigate to the project directory:
    ```bash
    cd health_model
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the prepare_data script to prepare the data
2. Run the train_model script to train the model
3. Run the test_model script to test the model

## Project Structure

- `data/`: Contains input datasets. After the prepare_data script is run two new files will be created from the original one: one for training, and one for testing.
- `model/`: Includes machine learning scripts and the model that will be saved after is trained.
- `results/`: Stores generated results and visualizations after testing the model.