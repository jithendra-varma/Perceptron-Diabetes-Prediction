
# Diabetes Prediction using Perceptron and Logistic Regression

This project involves predicting diabetes using the Pima Indians Diabetes dataset. We experimented with different machine learning models, including the Perceptron, GridSearch-optimized Perceptron, and Logistic Regression. The project is structured to showcase model performance comparisons across multiple evaluation metrics like recall, precision, F1 score, and accuracy.

## Project Overview

### Dataset
The Pima Indians Diabetes dataset is used for the prediction. It contains information about several medical predictor variables such as Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age, along with the target variable `Outcome`, which indicates whether the individual has diabetes or not.

### Models Implemented
1. **Normal Perceptron**: A simple Perceptron model without hyperparameter tuning.
2. **GridSearch Perceptron**: A Perceptron model with hyperparameter tuning using GridSearchCV.
3. **Perceptron with Reduced Features**: A Perceptron model trained on a reduced set of features (excluding Blood Pressure and Skin Thickness).
4. **Logistic Regression**: A standard logistic regression model used for comparison.

### Evaluation Metrics
The models are evaluated using the following metrics:
- **Recall**
- **Precision**
- **F1 Score**
- **Accuracy**

Additionally, confusion matrices are generated for visualizing the true positives, true negatives, false positives, and false negatives.

## Results Summary

| Model                     | Recall (%) | Precision (%) | F1 Score (%) | Accuracy (%) |
|----------------------------|------------|----------------|--------------|--------------|
| Normal Perceptron           | 67.71      | 75.00          | 71.21        | 73.38        |
| GridSearch Perceptron       | 72.88      | 80.00          | 76.27        | 76.62        |
| Perceptron (Reduced Features)| 71.19     | 76.92          | 73.94        | 75.32        |
| Logistic Regression         | 77.97      | 81.97          | 79.91        | 79.87        |

### Conclusion
The Logistic Regression model performed the best overall in terms of recall, precision, F1 score, and accuracy. The Perceptron models performed reasonably well, with improvements seen when tuning hyperparameters and reducing features.

## Visualizations

- **Histograms**: Display the distribution of each feature.
- **Correlation Heatmap**: Shows the correlation between different features.
- **Confusion Matrices**: Visualizes model performance in terms of correct and incorrect predictions.
- **Comparison Plots**: Shows a side-by-side comparison of recall, precision, F1 score, and accuracy across all models.

## Project Structure

```
.
├── diabetes_scale.txt       # Dataset used for training and testing
├── code.ipynb      # Main code for the project
├── README.md                # Project documentation
└── results                  # Folder to store results and figures
```


## Dependencies
- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
