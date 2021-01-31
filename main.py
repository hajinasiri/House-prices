import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
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
#Converting non-numerical columns to numerical
st_columns = train_data.select_dtypes(include= 'object')
st_columns = st_columns.columns
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for column in st_columns:
    enc.fit(train_data[column])
    train_data[column] = enc.transform(train_data[column])
    test_data[column] = enc.transform(test_data[column])

# sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()
# splitting train data intor training and testing set
y = train_data["SalePrice"]
X = train_data.drop('SalePrice',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)