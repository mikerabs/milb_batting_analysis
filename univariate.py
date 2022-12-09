import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("forPairPlot.csv")
X = data.iloc[:,3:25]  #independent variable columns
y = data.iloc[:,-1]    #target variable column (price range)


#print(X)
#print(y)
#extracting top 10 best features by applying SelectKBest class
bestfeatures = SelectKBest(score_func=chi2, k=10)
#wRAA giving trouble because of negative values
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
 
#concat two dataframes
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #printing 10 best features
