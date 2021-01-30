import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
df = train_data
bf = train_data
df = df.fillna(df.mean())
#create a dictionay of most frequent value in each column
def clean(data):
    #This function takes a dataframe as its variable and cleans it
    #Dropping columns with missing more than half of their data
    data = data.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
    # repalceing missing values from numerical columns with mean of the column
    data = data.fillna(data.mean())
    #replacing missing values from non-numerical columns with the most frequent value in the column
    data = data.fillna(data.mode().iloc[0])
    return data
train_data = clean(train_data)
test_data = clean(test_data)
ctrain = train_data.count()
ctest = test_data.count()
# sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()

t = train_data[train_data.columns[1:]].corr()['LotFrontage'][:-1]
average = train_data.mean()


