import pandas as pd
import numpy as np

# Generate a simple dataset
np.random.seed(42)
data_size = 100
feature1 = np.random.rand(data_size) * 100
feature2 = np.random.rand(data_size) * 100
target = 3.5 * feature1 + 2.5 * feature2 + np.random.randn(data_size) * 10

# Create a DataFrame
data = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'target': target})

# Save the dataset to a CSV file
data.to_csv('data/dataset.csv', index=False)
