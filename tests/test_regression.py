import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class TestRegression(unittest.TestCase):
    def test_regression(self):
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
        
        # Check if MSE is within an acceptable range
        self.assertLess(mse, 100, "Mean Squared Error is too high")

if __name__ == '__main__':
    unittest.main()
