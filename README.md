# ML_Project1_regression
## Multi-Variable Regression for House Price Prediction
### Overview
This project demonstrates how to build a machine learning model to predict house prices using multiple variables. The dataset used for training is based on the California housing data, and different models are tested to find the best one for the prediction.
1. Importing Libraries
   ```python
    import numpy as np
    import pandas as pd

2. Loading the Data
The dataset is loaded using pandas read_csv function:

    ```python
     data = pd.read_csv('/content/sample_data/california_housing_test.csv')
