# ML_Project1_regression
## Multi-Variable Regression for House Price Prediction
### Overview
This project demonstrates how to build a machine learning model to predict house prices using multiple variables. The dataset used for training is based on the California housing data, and different models are tested to find the best one for the prediction.
#### 1. Importing Libraries
   ```python
    import numpy as np
    import pandas as pd

2. Loading the Data
The dataset is loaded using pandas read_csv function:

   ```python
     data = pd.read_csv('/content/sample_data/california_housing_test.csv')

3. Inspecting the Data
The first two rows of the dataset are displayed:

   ```python
    data.head(2)
   
4. Checking for Missing Data
We check for any missing values in the dataset:

python
Copy
Edit
data.isnull().sum()
5. Correlation Analysis
We analyze the correlation of features with the target variable (median_house_value):

python
Copy
Edit
data.corr()['median_house_value']
6. Dropping Irrelevant Features
We drop longitude, latitude, and population as they are not essential for our prediction model:

python
Copy
Edit
data = data.drop(columns=['longitude','latitude','population'])
7. Shape of the Data
Check the number of rows and columns:

python
Copy
Edit
data.shape  # (3000, 6)
8. Defining Features and Labels
We split the data into features (x) and labels (y):

python
Copy
Edit
x = data.drop('median_house_value', axis=1)
y = data['median_house_value']
9. Splitting the Data into Training and Test Sets
We use train_test_split from sklearn.model_selection to divide the dataset into training and testing data:

python
Copy
Edit
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
10. Model Selection and Evaluation
We create a function to evaluate different models and find the best one based on accuracy:

python
Copy
Edit
def model_acc(model):
    model.fit(x_train, y_train)    # Training the model
    acc = model.score(x_test, y_test)  # Accuracy on test data
    print(str(model) + '-->' + str(acc))
11. Linear Regression Model
We test Linear Regression and print the accuracy:

python
Copy
Edit
from sklearn.linear_model import LinearRegression
lr = LinearRegression()  # Creating an object
model_acc(lr)  # Evaluating the model
Output:

scss
Copy
Edit
LinearRegression()-->0.5158794635875996
12. Lasso Regression Model
We test the Lasso model:

python
Copy
Edit
from sklearn.linear_model import Lasso
ls = Lasso()  # Creating an object
model_acc(ls)  # Evaluating the model
Output:

scss
Copy
Edit
Lasso()-->0.515879579371544
13. Decision Tree Regressor Model
We test the Decision Tree Regressor model:

python
Copy
Edit
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()  # Creating an object
model_acc(dt)  # Evaluating the model
Output:

scss
Copy
Edit
DecisionTreeRegressor()-->0.26880688251390705
14. Random Forest Regressor Model
We test the Random Forest Regressor model, which shows the highest accuracy so far:

python
Copy
Edit
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()  # Creating an object
model_acc(rf)  # Evaluating the model
Output:

scss
Copy
Edit
RandomForestRegressor()-->0.5693411202507745
15. Hyperparameter Tuning
We perform hyperparameter tuning using GridSearchCV to optimize the Random Forest model:

python
Copy
Edit
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [10, 50, 100], 'criterion': ['squared_error', 'absolute_error', 'poisson']}
grid_obj = GridSearchCV(estimator=rf, param_grid=parameters)
grid_fit = grid_obj.fit(x_train, y_train)

best_model = grid_fit.best_estimator_
best_model
After tuning, the best model is obtained, but the accuracy remains similar:

python
Copy
Edit
best_model.score(x_test, y_test)
Output:

Copy
Edit
0.5691648611795728
16. Saving the Model with Pickle
We use pickle to save the best model to a file:

python
Copy
Edit
import pickle
with open('predictor.pickle', 'wb') as file:
    pickle.dump(best_model, file)
17. Predicting House Value
We use the trained model to predict the price of a house given specific feature values:

python
Copy
Edit
best_model.predict([[2, 10, 3, 15, 1]])
Output:

scss
Copy
Edit
array([205200.18])
Conclusion
In this project, we tested several machine learning models, with Random Forest Regressor providing the best accuracy (around 0.569). We also explored hyperparameter tuning to optimize the model, and the model was then saved for future use.
