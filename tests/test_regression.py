import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class TestRegressionModel(unittest.TestCase):

    def setUp(self):
        # Create a dataset within the program
        np.random.seed(42)
        data_size = 100
        feature1 = np.random.rand(data_size) * 100
        feature2 = np.random.rand(data_size) * 100
        target = 3.5 * feature1 + 2.5 * feature2 + np.random.randn(data_size) * 10

        # Create a DataFrame with the target variable
        self.data_with_target = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'target': target})

        # Split the dataset into features and target variable
        self.X = self.data_with_target[['feature1', 'feature2']]
        self.y = self.data_with_target['target']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Initialize and train the regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def test_model_training(self):
        # Predict on the testing set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred)

        # Check if the model is trained and predictions are made
        self.assertIsNotNone(y_pred, "Model did not predict any values")
        
        # Check if Mean Squared Error is within an acceptable range
        self.assertLess(mse, 10.0, "Mean Squared Error is too high")

    def test_prediction_on_new_data(self):
        # Create another dataset without the target variable for prediction
        data_without_target = pd.DataFrame({'feature1': np.random.rand(20) * 100, 'feature2': np.random.rand(20) * 100})

        # Predict the target variable for the new dataset
        predicted_target = self.model.predict(data_without_target)
        data_without_target['predicted_target'] = predicted_target

        # Check if predictions are made on the new dataset
        self.assertTrue('predicted_target' in data_without_target.columns, "Predicted target column not found in new dataset")
        self.assertEqual(len(predicted_target), 20, "Number of predictions does not match the number of samples in the new dataset")

if __name__ == '__main__':
    unittest.main()
