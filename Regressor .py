
#%% 

# https://www.kaggle.com/data13/machine-learning-model-to-predict-app-rating-94#Data-Exploration-and-Cleaning

# Please add your own input lines , so that when we push it , we can all use our own  
#%% [markdown]
#Sarah 
import os
# dirpath = os.getcwd() 
# print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='C:/Users/sjg27/OneDrive/Documents/GWU Data Science/Fall 19/DATS 6103 Intro to DM/Project/googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'C:/Users/sjg27/OneDrive/Documents/GWU Data Science/Fall 19/DATS 6103 Intro to DM/Project/googleplaystore_user_reviews.csv'
userdata = pd.read_csv(filepath2)

#%% [markdown]
#Ayush
import os
# dirpath = os.getcwd() 
# print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore_user_reviews.csv'
userdata = pd.read_csv(filepath2)

#%% [markdown]
##Kartik 
import os
# dirpath = os.getcwd() 
# print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore_user_reviews.csv'
userdata = pd.read_csv(filepath2)


#%% [markdown]
#Trinh
import os
import pandas as pd
dirpath = os.getcwd() 
#print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'googleplaystore_user_reviews.csv'
userdata = pd.read_csv(filepath2)

#%%[markdown]

#################################################
#################### BEGIN ######################
#################################################



#%% 
# ## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')

#%%
# Standard quick checks
def dfChkBasics(dframe): 
  cnt = 1  
  try:
    print(str(cnt)+': info(): ')
    print(dframe.info())
  except: pass

  cnt+=1
  print(str(cnt)+': describe(): ')
  print(dframe.describe())

  cnt+=1
  print(str(cnt)+': dtypes: ')
  print(dframe.dtypes)

  try:
    cnt+=1
    print(str(cnt)+': columns: ')
    print(dframe.columns)
  except: pass

  cnt+=1
  print(str(cnt)+': head() -- ')
  print(dframe.head())

  cnt+=1
  print(str(cnt)+': shape: ')
  print(dframe.shape)

  # cnt+=1
  # print(str(cnt)+': columns.value_counts(): ')
  # print(dframe.columns.value_counts())

def dfChkValueCnts(dframe):
  cnt = 1
  for i in dframe.columns :
    print(str(cnt)+':', i, 'value_counts(): ')
    print(dframe[i].value_counts())
    cnt +=1
#%%
# Do a quick check
dfChkBasics(pdata)
dfChkValueCnts(pdata)




#%%

#View entire data set 
pdata.sample(10)
#%%
# pdata.describe
pdata.shape
#%%
import re
import sys

import time
import datetime

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#%%
df = pdata 
# %%
%matplotlib inline
#%%
sns.set(style='darkgrid')
sns.set_palette('PuBuGn_d')

#%%

# Checking the data type of the columns
pdata.info()

#%%[markdown]
# #### The dataset has 10,841 records and 13 columns, all of them are object types except the target column (Rating) which is float


#%%[markdown]
#  #### Exploring missing data and checking if any has NaN values
plt.figure(figsize=(7, 5))
sns.heatmap(df.isnull(), cmap='viridis')
df.isnull().any()

# %%[markdown]
# #### Looks like there are missing values in "Rating", "Type", "Content Rating" and " Android Ver". But most of these missing values in Rating column.
df.isnull().sum()

# %%[markdown]
# #### The best way to fill missing values might be using the median instead of mean.
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

# #### Before filling null values we have to clean all non numerical values & unicode charachters 
replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]

for i in replaces:
	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))

regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']

for j in regex:
	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)
df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].median())

#%%[markdown]
# ### Count the number of unique values in category column 
df['Category'].unique()

#%%
df['Genres'].unique().shape



# %%[markdown]
# Check the record  of unreasonable value which is 1.9
i = df[df['Category'] == '1.9'].index
df.loc[i]

# %%
#It's obvious that the first value of this record is missing (App name) and all other values are respectively propagated backward starting from "Category" towards the "Current Ver"; and the last column which is "Android Ver" is left null. It's better to drop the entire recored instead of consider these unreasonable values while cleaning each column!

# %%
# Drop this bad column
df = df.drop(i)

# %%
# Removing NaN values
df = df[pd.notnull(df['Last Updated'])]
df = df[pd.notnull(df['Content Rating'])]

#%%[markdown]
# # Categorical Data Encoding

# #### Many machine learning algorithms can support categorical values without further manipulation but there are many more algorithms that do not. We need to make all data ready for the model, so we will convert categorical variables (variables that stored as text values) into numircal variables.

# %%
print(pdata['App'].head() , "\n \n  AFTER ENCODING : \n ")

le = preprocessing.LabelEncoder()
df['App'] = le.fit_transform(df['App'])
# This encoder converts the values into numeric values
print(df['App'].head())


# %%[markdown]
# ### Category features encoding
category_list = df['Category'].unique().tolist() 
category_list = ['cat_' + word for word in category_list]
df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)

# %%[markdown]
# ### Genres features encoding
le = preprocessing.LabelEncoder()
df['Genres'] = le.fit_transform(df['Genres'])

# %%[markdown]
# ### Encode Content Rating features
le = preprocessing.LabelEncoder()
df['Content Rating'] = le.fit_transform(df['Content Rating'])

# %%[markdown]
# ### Price Cleaning
df['Price'] = df['Price'].apply(lambda x : x.strip('$'))

# %%[markdown]
# ### Installs Cleaning
df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))

# %%[markdown]
# ### Type encoding
df['Type'] = pd.get_dummies(df['Type'])

# %%[markdown]
# ### The above line drops the reference column and just keeps only one of the two columns as retaining this extra column does not add any new information for the modeling process, this line is exactly the same as setting drop_first parameter to True.





# %%[markdown]
# ### Last Updated encoding
df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))


# %%[markdown]

# ### Convert kbytes to Mbytes 
k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
df.loc[k_indices,'Size'] = converter

# %%[markdown]

# #### This can be done by selecting all k values from the "Size" column and replace those values by their corresponding M values, and since k indices belong to a list of non-consecutive numbers, a new dataframe (converter) will be created with these k indices to perform the conversion, then the final values will be assigned back to the "Size" column.

# %%[markdown]
# Size cleaning
df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
df[df['Size'] == 'Varies with device'] = 0
df['Size'] = df['Size'].astype(float)

# %%[markdown]

# # Evaluation Procedure 
# ### In this section shows how k-nearest neighbors and random forests can be used to predict app ratings based on the other matrices. First, the dataset has to separate into dependent and independent variables (or features and labels). Then those variables have to split into a training and test set.

# ### During training stage we give the model both the features and the labels so it can learn to classify points based on the features.

# %%
# Split data into training and testing sets
features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']
features.extend(category_list)
X = df[features]
y = df['Rating']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

# %%[markdown]
 # ## The above script splits the dataset into 85% train data and 25% test data.


# %%[markdown]
# # KNN Model 
# ### The k-nearest neighbors algorithm is based around the simple idea of predicting unknown values by matching them with the most similar known values. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset — its "nearest neighbors".




# %%
# Look at the 15 closest neighbors
model = KNeighborsRegressor(n_neighbors=15)

# %%
# Find the mean accuracy of knn regression using X_test and y_test
model.fit(X_train, y_train)


# %%# Calculate the mean accuracy of the KNN model
accuracy = model.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 3)) + '%'

#%%
y_pred = model.predict(X_test)


# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_pred)
# from sklearn.metrics import classification_report
# classification_report(y_test, y_pred)

from sklearn.metrics import explained_variance_score
print("Explained Variance",explained_variance_score(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print("Mean_squared_error",mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
print("r Square error",r2_score(y_test, y_pred))


# %%
# Try different numbers of n_estimators - this will take a minute or so
n_neighbors = np.arange(1, 50, 1)
scores = []
rmse_val = []
for n in n_neighbors:
    model.set_params(n_neighbors=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    error = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
    rmse_val.append(error)
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("R2 Score")
plt.plot(n_neighbors, scores)

#elbow curve
curve = pd.DataFrame(rmse_val)
curve.plot()
plt.title("RMSE Value's")
plt.xlabel("Number of Neighbors K")
plt.ylabel("RMSE")
print("Ideal K value at the elbow at K=10")

# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/









#%%
n_neighbors = 5

for i, weights in enumerate(['uniform']):
    knn = KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X_train, y_train).predict(X_test)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X_test, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.tight_layout()
plt.show()



#%%
from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X_train)

X_train1 = pd.concat([X_train,y_train],axis=1)

xdata = X_train1[['Rating', 'Reviews', 'Installs', 'Price','Size','Genres','Type']]
#%%

km_xdata = KMeans( n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 3 clusters
index1 = 4#5 #4 
index2 = 0#0

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=25, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=25, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=25, c='lightblue', marker='v', edgecolor='black', label='cluster 3',alpha=0.1)

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
km_xdata = KMeans( n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 3 clusters
index1 = 5 #5 #4 
index2 = 0#0

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=25, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=25, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=25, c='lightblue', marker='v', edgecolor='black', label='cluster 3')

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()
#%%
# 4 clusters
km_xdata = KMeans( n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 4 clusters
index1 = 4 #Size
index2 = 0 #Rating

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=25, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=25, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=25, c='lightblue', marker='v', edgecolor='black', label='cluster 3')

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=25, c='pink', marker='p', edgecolor='black', label='cluster 4')

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# 4 clusters
km_xdata = KMeans( n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 4 clusters
index1 = 5 #Genre
index2 = 0 #Rating

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=25, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=25, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=25, c='lightblue', marker='v', edgecolor='black', label='cluster 3')

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=25, c='pink', marker='p', edgecolor='black', label='cluster 4')

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()
#%%

# %%[markdown]

# ### Random Forest Model
# ### The RandomForestRegressor class of the sklearn.ensemble library is used to solve regression problems via random forest. The most important parameter of the RandomForestRegressor class is the n_estimators parameter. This parameter defines the number of trees in the random forest.

# %%

print(X_train.sample (5))
#%%  
model = RandomForestRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so


estimators = np.arange(10, 200, 10)
scores = []

for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
results = list(zip(estimators,scores))
results

# %%
predictions = model.predict(X_test)
#%%
'Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions)
#%%
'R^2', metrics.r2_score(y_test,predictions)

#%% 
'Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)


# %%
'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))


# %%
from sklearn import cross_validation
from sklearn.model_selection import KFold 
# from sklearn.cross_validation import cross_val_score,cross_val_predict
from sklearn import metrics

# %%
# Perform 6-fold cross validation
from sklearn.model_selection import cross_val_score

estimators = np.arange(2, 20, 1)
mscores = []
for i in estimators:
  scores = cross_val_score(model,X_test, y_test, cv=i)
  # print ("Cross-validated scores:", scores)
  print("The mean from CV = ",i," :  ", np.mean(scores))
  mscores.append(np.mean(scores))

plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("mean CV score")
plt.plot(estimators, mscores)
results = list(zip(estimators,scores))
results

# %%

# Gradient Boosting Regression 

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#           'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor()
estimators = np.arange(10, 600, 50)
scores = []

for n in estimators:
    clf.set_params(n_estimators=n)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no: of Estimators")
plt.ylabel("R2 Score")
plt.plot(estimators, scores)
results = list(zip(estimators,scores))
results
# clf.fit(X_train, y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))

# print(clf.score(X_test,y_test)*100)
# print("MSE: %.8f" % mse)

# %%
