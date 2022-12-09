import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("forPairPlot.csv")
print(df.head())

plt.subplots(figsize = (25,8))
heatmap = sns.heatmap(df.corr(),vmin=-1, vmax = 1,annot = True, cmap="RdYlGn")
plt.savefig("heatmap.svg")




#g = sns.FacetGrid(df,col = "Level")
#wOBAplot = g.map(sns.lmplot, "BABIP", "wOBA")

#wOBAplot2 = sns.lmplot(x = "wRC", y = "wOBA", hue = "Level",col = "Level", col_wrap =3,robust = False, data = df, order = 1, ci = 95)

# Visualizing multicollinearity between independent features using a heatmap  
  
corr = df.corr()  
print('Pearson correlation coefficient matrix for each independent variable: \n', corr)  
  
# Masking the diagonal cells   
masking = np.zeros_like(corr, dtype = np.bool)  
np.fill_diagonal(masking, val = True)  
  
# Initializing a matplotlib figure  
figure, axis = plt.subplots(figsize = (12, 9))  
  
# Generating a custom colormap  
c_map = sns.diverging_palette(223, 14, as_cmap = True, sep = 100)  
c_map.set_bad('grey')  
  
# Displaying the heatmap with the masking and the correct aspect ratio  
sns.heatmap(corr, mask = masking, cmap="RdYlGn", vmin = -1, vmax = 1, center = 1, linewidths = 1)  
figure.suptitle('Heatmap visualizing Pearson Correlation Coefficient Matrix', fontsize = 14)  
axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

plt.savefig("heatmapcorr.svg")

#wOBAplot = sns.pairplot(data = df, hue = "Level",height = 2)

#plt.savefig("wOBAplot2.svg")

#OBP vs O-Swing% order 2

