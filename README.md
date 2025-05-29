Dataset Description
The dataset contains information about students' study habits and performance. It includes the following features:

Hours Studied — Number of hours the student spent studying.

Previous Scores — Scores from previous tests.

Extracurricular Activities — Whether the student participates in extracurricular activities or sports (Yes/No).

Sleep Hours — Average number of hours the student sleeps per day.

Sample Question Papers Practiced — Number of practice test papers completed by the student.

The target variable is the final test score (or another measure of academic performance).

This dataset is used to analyze how different factors affect students’ performance and to build predictive models estimating final scores based on study and lifestyle patterns.

This project implements linear regression models to predict a target variable based on a set of features. It includes:

* **Linear Regression**
* **Lasso Regression**
* **Ridge Regression**

The data is preprocessed with feature scaling (StandardScaler), and hyperparameters for Lasso and Ridge are optimized using GridSearchCV with cross-validation.

---

##Features

* Data preprocessing and feature scaling
* Training and evaluation of Linear, Lasso, and Ridge regression models
* Hyperparameter tuning (alpha) for Lasso and Ridge using GridSearchCV
* Model performance evaluation with metrics: MSE, R², MAE
* Visualization of results (optional, if implemented)

---

## Installation

Requires Python 3.x and the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

##Usage

1. **Prepare your data** 
Load your dataset and split into training and testing sets.

2. **Train models** 
Fit LinearRegression, Lasso, and Ridge models on the training data.

3. **Hyperparameter tuning** 
Use GridSearchCV with a pipeline (StandardScaler + model) to find the best alpha parameter for Lasso and Ridge.

4. **Evaluate models** 
Calculate metrics on the test data: MSE, R², and MAE.

Example code snippet for hyperparameter tuning:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

pipeline_lasso = Pipeline([ 
('scaler', StandardScaler()), 
('lasso', Lasso())
])

param_grid_lasso = {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_lasso = GridSearchCV(pipeline_lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
grid_lasso.fit(x_train, y_train)

print("Best alpha for Lasso:", grid_lasso.best_params_)
print("Best negative MSE:", grid_lasso.best_score_)
```

---

##Results

* Best alpha for both Lasso and Ridge: **0.001**
* Performance on test set: 

* Lasso MSE: \~4.08, R²: \~0.989, MAE: \~1.61 
* Ridge MSE: \~4.08, R²: \~0.989, MAE: \~1.61

Models demonstrate good predictive accuracy and stability after hyperparameter tuning.

---

##Notes

* Feature scaling is crucial before applying regularized regression.
* GridSearchCV helps prevent overfitting by selecting optimal regularization strength.
* Adjust alpha grid range depending on your data characteristics.

