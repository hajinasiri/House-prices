import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
data = [train_data,test_data]
def clean(data):
    #This function takes a dataframe as its variable and cleans it
    data = data.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
    data['FireplaceQu'] = data['FireplaceQu'].map({'Ex': 4, 'Gd': 3,'TA':2,
                                                   'Fa':1,'Po':0})
    data['FireplaceQu'].fillna(2, inplace=True)
    return data
train_data = clean(train_data)
test_data = clean(test_data)
c = train_data.count()
# sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()

t = train_data[train_data.columns[1:]].corr()['LotFrontage'][:-1]
print(t)
