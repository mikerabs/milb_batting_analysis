'''The f_regression function takes a two-dimensional array or matrix containing the independent variables 
and a one-dimensional array containing the dependent variable as arguments, and returns the F-scores and 
p-values for each independent variable. The F-score and p-value for an independent variable indicate how 
strongly that variable is associated with the dependent variable, with higher F-scores and lower p-values indicating a stronger association.'''

import pandas as pd
from sklearn.feature_selection import f_regression

# Read the independent and dependent variables from a .csv file
df = pd.read_csv("forFeatureSelect.csv")

# Select the columns containing the independent variables
X = df[df.columns[:-1]].values

# Select the column containing the dependent variable
y = df[df.columns[-1]].values

# Perform the statistical test to determine the relevance of each independent variable
f_scores, p_values = f_regression(X, y)

# Print the F-scores and p-values for each independent variable
#for i, (f_score, p_value) in enumerate(zip(f_scores, p_values)):
 #   print(f"Variable x{i+1}: F-score={f_score:.3f}, p-value={p_value:.3f}")
for column, (f_score, p_value) in zip(df.columns[:-1], zip(f_scores, p_values)):
    print(f"{column}: F-score={f_score:.3f}, p-value={p_value:.3f}")
