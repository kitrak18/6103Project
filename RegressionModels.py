#ML to Prediction Models
#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
%matplotlib inline
#%%

import os
import pandas as pd
dirpath = os.getcwd() 
#print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='googleplaystore.csv'
data1 = pd.read_csv(filepath1)

filepath2 = 'googleplaystore_user_reviews.csv'
data = pd.read_csv(filepath2)
#%%

data1.head()
data1.shape
data1.columns
#%%
#Group By Different Variables
dt_ctg=data1.groupby('Category',as_index=False)['Rating'].mean()
dt_ctg.head()

dt_sz=data1.groupby('Size',as_index=False)['Rating'].mean()
dt_sz.head(10)

dt_in=data1.groupby('Installs',as_index=False)['Rating'].mean()
dt_in.head(6)

dt_gn=data1.groupby('Genres',as_index=False)['Rating'].mean()
dt_gn.head(10)

dt_tp=data1.groupby('Type',as_index=False)['Rating'].mean()
dt_tp.head()

dt_prc=data1.groupby('Price',as_index=False)['Rating'].mean()
dt_prc.head(10)

dt_cr=data1.groupby('Content Rating',as_index=False)['Rating'].mean()
dt_cr.head(10)

dt_av=data1.groupby('Android Ver',as_index=False)['Rating'].mean()
dt_av.head()
#%%
# Data Visualizations

sns.countplot(x='Type',data=data1)

sns.barplot(x='Type', y='Rating', data=data1)

plt.figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
#%%
# specify hue="categorical_variable"
sns.barplot(x='Content Rating', y='Rating', hue="Type", data=data1, estimator=np.median)
plt.show()

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
#%%
# specify hue="categorical_variable"
sns.boxplot(x='Content Rating', y='Rating', hue="Type", data=data1)
plt.show()

plt.figure(figsize=(16,8))
#%%
sns.countplot(y='Category',data=data1)
plt.show()

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k')
# specify hue="categorical_variable"
#%%
sns.barplot(y='Category', x='Rating', hue="Type", data=data1, estimator=np.median)
plt.show()

sns.countplot(x='Content Rating',data=data1)
#%%

sns.boxplot(x='Content Rating', y='Rating', data=data1)
#%%
sns.barplot(x='Content Rating', y='Rating', data=data1)
#%%
plt.figure(figsize=(7,14))
sns.barplot(y='Installs', x='Rating', data=data1)
plt.show()
#%%
plt.figure(figsize=(10, 25))
sns.barplot(y='Android Ver', x='Rating', data=data1)
plt.show()
#%%
plt.figure(figsize=(8, 15))
sns.countplot(y='Rating',data=data1)
plt.show()
#%%
#Data Cleaning

data1[data1['Rating'] == 19]

data1[10470:10475]

data1.iloc[10472,1:] = data1.iloc[10472,1:].shift(1)
data1[10470:10475]

data1.isnull().sum().sum()

total=data1.isnull().sum()
percent = (data1.isnull().sum()/data1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(13)

data1.dropna(inplace=True)
data1.shape
#%%
#Creating Dummy Variables

data1.head()

catgry=pd.get_dummies(data1['Category'],prefix='catg',drop_first=True)
typ=pd.get_dummies(data1['Type'],prefix='typ',drop_first=True)
cr=pd.get_dummies(data1['Content Rating'],prefix='cr',drop_first=True)
frames=[data1,catgry,typ,cr]
data1=pd.concat(frames,axis=1)
data1.drop(['Category','Installs','Type','Content Rating'],axis=1,inplace=True)
data1.drop(['App','Size','Price','Genres','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)

data1.head()
#%%
#Feature Selection

X=data1.drop('Rating',axis=1)
y=data1['Rating'].values
y=y.astype('int')
#%%
#Train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#%%
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#%%
#LogisticRegression
lr_c=LogisticRegression(random_state=0)
lr_c.fit(X_train,y_train)
lr_pred=lr_c.predict(X_test)
lr_cm=confusion_matrix(y_test,lr_pred)
lr_ac=accuracy_score(y_test, lr_pred)
print('LogisticRegression_accuracy:',lr_ac)

plt.figure(figsize=(10,5))
plt.title("lr_cm")
sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.show()
#%%
#Decision Tree Classifier

dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=120)
dtree_c.fit(X_train,y_train)
dtree_pred=dtree_c.predict(X_test)
dtree_cm=confusion_matrix(y_test,dtree_pred)
dtree_ac=accuracy_score(dtree_pred,y_test)

plt.figure(figsize=(10,5))
plt.title("dtree_cm")
sns.heatmap(dtree_cm,annot=True,fmt="d",cbar=False)
print('DecisionTree_Classifier_accuracy:',dtree_ac)
#%%
#SVM regressor
svc_r=SVC(kernel='rbf')
svc_r.fit(X_train,y_train)
svr_pred=svc_r.predict(X_test)
svr_cm=confusion_matrix(y_test,svr_pred)
svr_ac=accuracy_score(y_test, svr_pred)

plt.figure(figsize=(10,5))
plt.title("svm_cm")
sns.heatmap(svr_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)
print('SVM_regressor_accuracy:',svr_ac)
#%%
#RandomForest
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
rdf_ac=accuracy_score(rdf_pred,y_test)

plt.figure(figsize=(10,5))
plt.title("rdf_cm")
sns.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
print('RandomForest_accuracy:',rdf_ac)
#%%
model_accuracy = pd.Series(data=[lr_ac,dtree_ac,svr_ac,rdf_ac], 
        index=['Logistic_Regression','DecisionTree_Classifier','SVM_regressor_accuracy','RandomForest'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accracy')



# %%
