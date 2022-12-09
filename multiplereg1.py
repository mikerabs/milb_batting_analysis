import pandas as pd
from statsmodels.formula.api import ols

# read in the data from a csv file and store it in a DataFrame
data = pd.read_csv('forModel.csv')

# specify the formula for the regression, where 'dependent_variable' is the name of the column containing
# the dependent variable and the other columns are the names of the independent variables
#formula = 'wOBA ~ Ldpercent + Gbpercent + Centpercent + Oppopercent'#using the sklearn recursive feature selection
formula = 'wOBA ~ OBP + BABIP + wRAA'

# fit the model using the formula and data
model = ols(formula, data).fit()

# print a summary of the model
print(model.summary())
