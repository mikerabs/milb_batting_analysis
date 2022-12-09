import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# read in the data from a csv file and store it in a DataFrame
data = pd.read_csv('forModel.csv')

# separate the dependent and independent variables
X = data[['wRCplus', 'wRAA', 'OBP', 'SLG', 'AVG', 'wRC','BABIP', 'ISO']]
y = data['wOBA']

# fit the model using the independent and dependent variables
model = LinearRegression().fit(X, y)

# make predictions using the model
y_pred = model.predict(X)

# calculate the r2 score, which measures the accuracy of the predictions
score = r2_score(y, y_pred)

# print the coefficients and intercept of the model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# print the r2 score
print('R2 Score:', score)
