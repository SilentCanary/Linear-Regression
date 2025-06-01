"""
@author: silent._.canary
"""
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['MedHouseVal']=data.target
df['bedrooms_per_room']=df['AveBedrms']/df['AveRooms']

corr=df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()

plt.scatter(df['Longitude'], df['MedHouseVal'], alpha=0.3)
plt.show()