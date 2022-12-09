import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# read in the data from a csv file and store it in a DataFrame
data = pd.read_csv('forModel.csv')

# separate the dependent and independent variables
X = data[['wRAA', 'OBP','BABIP']]
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

# create a DataFrame with the actual and predicted values
results = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

# create a figure object and set its size
fig = plt.figure()
fig.set_size_inches(12, 6)

# plot the actual and predicted values using seaborn
sns.lineplot(data=results)
sns.set(style='darkgrid', palette='muted')

plt.xlabel("BABIP + OBP + wRAA")
plt.ylabel("wOBA predictions")
plt.title("wOBA ~ BABIP + OBP + wRAA")

# show the plot
plt.savefig("multiplereg.svg")




# create a figure with 3 rows and 2 columns of subplots
fig, ax = plt.subplots(nrows=3, ncols=2)

# adjust the size of the subplots
fig.set_size_inches(12, 6)

# plot the actual and predicted values in each subplot
for i in range(3):
    for j in range(2):
        ax[i, j].plot(y[i*200+j*1000:i*200+(j+1)*1000], color='blue', label='Actual')
        ax[i, j].plot(y_pred[i*200+j*1000:i*200+(j+1)*1000], color='red', label='Predicted')
        ax[i, j].legend(loc='upper left')

plt.savefig('multipleregsub.svg')
