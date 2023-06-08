# Intensity-based Speaker Classification Project

This project aims to classify speaker intensity based on various features using a logistic regression model with feature selection and scaling techniques. The goal is to provide a tool for analyzing and predicting speaker intensity levels based on specific input features.

## Installation

To use this project, follow the steps below:

1. Clone the repository to your local machine:

`git clone https://github.com/your-username/intensity-classification.git`

2. Install the required dependencies by running the following command in your terminal:

`pip install -r requirements.txt`

## Usage

1. Ensure that the intensity.csv file is present in the project directory. This file contains the dataset used for training and testing the model.

2. Open the Main.ipynb notebook file using Jupyter Notebook or JupyterLab.

`jupyter-notebook Main.ipynb`

3. Inside the notebook, execute the cells in sequential order to run the project.

- The notebook will load the dataset from intensity.csv and split it into training and testing sets.
- It will then apply feature scaling using the StandardScaler to normalize the feature values.
- Feature selection will be performed using the SelectKBest method with the f_classif scoring function to select the most relevant features.
- A logistic regression model will be trained using the selected features.
- The trained model will be evaluated on the testing set, and a classification report will be generated, displaying metrics such as precision, recall, and F1-score for each class.

This script performs the following steps:
- Loads the dataset from `intensity.csv`.
- Splits the dataset into training and testing sets.
- Applies feature scaling using `StandardScaler`.
- Performs feature selection using `SelectKBest` with the `f_classif` scoring function.
- Trains a logistic regression model with the selected features.
- Evaluates the model on the testing set and generates a classification report.

3. The classification report is printed to the console, providing metrics such as precision, recall, and F1-score for each class.

## Results

The classification report provides an in-depth evaluation of the model's performance on the testing set. It includes metrics such as precision, recall, and F1-score for each class, allowing you to assess how well the model performs for different intensity levels. The accuracy, macro average, and weighted average metrics give an overall view of the model's performance across all classes.

Please note that the logistic regression model's convergence may generate a `ConvergenceWarning` due to the `max_iter` parameter. Adjusting this parameter may be necessary depending on the dataset and specific requirements.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](LICENSE).

You are free to:

- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material.

Under the following terms:

- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial: You may not use the material for commercial purposes.
- NoDerivatives: If you remix, transform, or build upon the material, you may not distribute the modified material.

Please note that this work is solely for learning purposes and may not be used, shared, or distributed without explicit permission from the author.
