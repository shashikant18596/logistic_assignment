import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import utils,preprocessing

data = sm.datasets.fair.load_pandas().data
pd.set_option("display.max_columns",None)
print(data)

#print(data.describe())
'''for columns in data.columns:
    sns.displot(data[columns],kde = True)
plt.show()

for columns in data:
    sns.boxplot(data=data[data.columns])
plt.show()'''
x = data.drop(columns = data[['affairs']],axis = 1)
y = data['affairs']
y = pd.to_numeric(y)
print(y)
y = preprocessing.LabelEncoder().fit_transform(y)

from sklearn.preprocessing import StandardScaler
data_scaled = StandardScaler().fit_transform(x)
print(data_scaled)

vif = pd.DataFrame()
vif['vif']= [variance_inflation_factor(data_scaled,i)for i in range(data_scaled.shape[1])]
print(vif)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_scaled,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=100000000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)
print(model.score(x_test,y_test))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

