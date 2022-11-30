import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("fangraphs-minor-league-leaders-advanced.csv")
print(df.head())

plt.subplots(figsize = (25,8))
heatmap = sns.heatmap(df.corr(),vmin=-1, vmax = 1,annot = True, cmap = 'BrBG')
plt.savefig("heatmap.svg")




#g = sns.FacetGrid(df,col = "Level")
#wOBAplot = g.map(sns.lmplot, "BABIP", "wOBA")

wOBAplot = sns.lmplot(x = "wRC", y = "wOBA", hue = "Level",col = "Level", col_wrap =3,robust = False, data = df, order = 1, ci = 95)

plt.savefig("wOBAplot.svg")

#OBP vs O-Swing% order 2

