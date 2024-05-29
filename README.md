# Basic Machine Learning Task: Regression

## Objective
The objective of this assignment is to write a Python program to perform a basic machine learning task using regression. You will train a regression model on a small dataset and make predictions.

## Instructions
1. Fork this repository to your GitHub account.
2. Clone the forked repository to your local machine.
3. Install the required libraries using `pip install -r requirements.txt`.
4. Run the script `data/generate_data.py` to create a small dataset.
5. Implement the regression model in `regression.py`.
6. Ensure your model passes the unit tests provided in `tests/test_regression.py`.
7. Commit your changes and push them to your forked repository.

## Requirements
- Use the libraries specified in `requirements.txt`.
- Your code should create a dataset, train a regression model, and evaluate it using Mean Squared Error (MSE).

## Example
Here is a basic structure of the `regression.py` script:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('data/dataset.csv')

# Split the dataset into features and target variable
X = data[['feature1', 'feature2']]
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
