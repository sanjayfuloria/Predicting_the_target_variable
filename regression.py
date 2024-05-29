import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Create a dataset within the program
np.random.seed(42)
data_size = 100
feature1 = np.random.rand(data_size) * 100
feature2 = np.random.rand(data_size) * 100
target = 3.5 * feature1 + 2.5 * feature2 + np.random.randn(data_size) * 10

# Create a DataFrame with the target variable
data_with_target = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'target': target})

# Split the dataset into features and target variable
X = data_with_target[['feature1', 'feature2']]
y = data_with_target['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)
print(f'The Predicted Values on Test Set: {y_pred}')

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Create another dataset without the target variable for prediction
data_without_target = pd.DataFrame({'feature1': np.random.rand(20) * 100, 'feature2': np.random.rand(20) * 100})

# Predict the target variable for the new dataset
predicted_target = model.predict(data_without_target)
data_without_target['predicted_target'] = predicted_target

# Output the dataset with the predicted target
print("New dataset with predicted target values:")
print(data_without_target)
