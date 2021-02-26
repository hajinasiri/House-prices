import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
test_data = pd.read_csv("test.csv")
index = test_data['Id']
train_data = pd.read_csv("train.csv")
original = train_data
train_data = train_data.loc[train_data['SalePrice'] <= 400000]#gets rid of the outlier prices
y = train_data["SalePrice"]
# sns.displot(y)#to plot the price distribution

#create a dictionay of most frequent value in each column
def clean(data):
    data['BsmtFinType1'] = data['BsmtFinType1'].replace('NA', 'Missing')
    data['BsmtQual'] = data['BsmtQual'].replace('NA', 'Missing')
    data['BsmtCond'] = data['BsmtCond'].replace('NA', 'Missing')
    data['BsmtExposure'] = data['BsmtExposure'].replace('NA', 'Missing')
    data['BsmtExposure'] = data['BsmtExposure'].replace('NA', 'Missing')

    #This function takes a dataframe as its variable and cleans it
    #Dropping columns with missing more than half of their data
    data = data.drop(['Id','Alley','PoolQC','Fence','MiscFeature','Street'],axis=1)
    data = data.fillna(data.mean())# repalce missing values from numerical columns with mean of the column
    data = data.fillna(data.mode().iloc[0])#replace missing values from non-numerical columns with the most frequent value in the column
    return data
c = clean(train_data)
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
# train_data = np.power(train_data, 0.3)
# test_data = np.power(test_data,0.3)

# sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()


X = train_data.drop('SalePrice',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)# split train data intor training and testing set
#Using linear regression algorithm
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X_train, y_train)  # perform linear regression
prediction = linear_regressor.predict(X_test)  # make predictions
coef = pd.DataFrame(linear_regressor.coef_,X_train.columns,columns=['Coef'])#create a dataframe from coefficiants to study

from sklearn import metrics
print(metrics.mean_absolute_error(y_test,prediction))
print(metrics.mean_squared_error(y_test,prediction))
#
linear_regressor.fit(X,y)#use all training data to train the model
result = linear_regressor.predict(test_data)#Predict test data's price
result = pd.DataFrame(result,index=index,columns=['SalePrice'])
result.index.names = ['Id']
result.to_csv('./house_price_result.csv')

# Finding features that have bigger than 0.2 correlation with SalePrice
corrmat = train_data.corr()
top_corr_features = corrmat.index
v = train_data[top_corr_features].corr()
v1 = v[(v['SalePrice'] >= 0.2) & (v['SalePrice'] < 1)]#smaller than 1 to get rid of the 'salePrice'
# as a feature
v2 = v[v['SalePrice'] <= -0.2]
v = v1.append(v2)
col = v.index

# Keeping the desired features to train the model
featured_train = X_train[col].copy()
featured_train = np.power(featured_train, 0.3)
featured_test = X_test[col].copy()
featured_test = np.power(featured_test, 0.3)
l = LinearRegression()  # create object for the class
l.fit(featured_train, y_train)  # perform linear regression
prediction = l.predict(featured_test)  # make predictions
print(metrics.mean_absolute_error(y_test,prediction))
print(metrics.mean_absolute_error(y_train,l.predict(featured_train)))

import seaborn as sns
#get correlations of each features in dataset
# corrmat = train_data.corr()
#
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g=sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()
print("----Tensorflow_____")
#tensorflow
t_train = X_train.values
t_test = X_test.values
t_y_train = y_train.values
t_y_test = y_test.values
#normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
t_train = scaler.fit_transform(t_train)
t_test = scaler.transform(t_test)
#creating neural network model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
model = Sequential()
model.add(Dense(74,activation='tanh',kernel_regularizer=regularizers.L1(0.001)))
model.add(Dense(55,activation='relu',kernel_regularizer=regularizers.L1(0.001)))
model.add(Dense(40,activation='relu',kernel_regularizer=regularizers.L1(0.001)))
model.add(Dense(30,activation='relu',kernel_regularizer=regularizers.L1(0.001)))
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam',loss='mse')
model.fit(x=t_train, y=t_y_train,validation_data=(t_test,t_y_test),batch_size=128,epochs=340)
loss = pd.DataFrame(model.history.history)
loss.plot()
plt.show()
#evaluate the model
print(model.evaluate(t_test,y_test,verbose=0))
test_prediction = model.predict(t_test)
test_prediction = pd.Series(test_prediction.reshape(473,))
pred_def = pd.DataFrame(y_test,columns=['Test True Y'])
pred_def = pd.concat([pred_def,test_prediction],axis=1)
pred_def.columns = ['Test True y', 'Model Predictions']
print(metrics.mean_absolute_error(t_y_test,test_prediction))
print(np.sqrt(metrics.mean_squared_error(t_y_test,test_prediction)))
from sklearn.metrics import explained_variance_score
print(explained_variance_score(t_y_test, test_prediction))