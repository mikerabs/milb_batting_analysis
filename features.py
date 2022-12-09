import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Read the data from the .csv file
df = pd.read_csv("forFeatureSelect.csv")

# Select the dependent and independent variables
X = df[df.columns[:-1]]#selects all variables not the last column
y = df[df.columns[-1]]#this is the last column, wOBA

#print(X)
#print(y)

# Create a linear regression model
model = LinearRegression()

# Use recursive feature elimination to select the independent variables
rfe = RFE(model, n_features_to_select=5)
X_selected = rfe.fit_transform(X, y)

# Print the selected independent variables
print(X.columns[rfe.support_])
