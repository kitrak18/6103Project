
#%% 
# ## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
#%%
# https://www.kaggle.com/data13/machine-learning-model-to-predict-app-rating-94#Data-Exploration-and-Cleaning

# Please add your own input lines , so that when we push it , we can all use our own  
#%% [markdown]
#Sarah 
import os
import pandas as pd
#%%
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
pdata.head()
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
import tensorflow as tf
print('tensorflow version : ', tf.__version__)

# default libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns
sns.set()
from sklearn import preprocessing
# for data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# matplotlib inline

# for classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

# for models evaluation
from sklearn.metrics import confusion_matrix, accuracy_score

#%%
df = pdata 
# %%
# %matplotlib inline
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

#%%
df = df.drop(columns = ['Category','Current Ver','Android Ver'])


#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled=pd.DataFrame(scaler.fit_transform(df))
df_scaled.columns=df.columns
df_scaled.index=df.index
df_scaled.describe()
# %%[markdown]

# # Evaluation Procedure 
# ### In this section shows how k-nearest neighbors and random forests can be used to predict app ratings based on the other matrices. First, the dataset has to separate into dependent and independent variables (or features and labels). Then those variables have to split into a training and test set.

# ### During training stage we give the model both the features and the labels so it can learn to classify points based on the features.
df = df_scaled
# %%
# Split data into training and testing sets
features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']
features.extend(category_list)
X = df[features]
y = df['Rating']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# %%[markdown]
 # ## The above script splits the dataset into 85% train data and 25% test data.


#%%[markdown]
# ## Neural Networks
# import eli5
# !pip install graphviz
# !pip install pydotplus
# from graphviz 
import keras 
import pydot
import pydotplus

#%%
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape = [X_train.shape[1]]),
  # tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),  
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='relu'),  
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),  
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')                     
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

#%%
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True,rankdir='TB')



history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test),batch_size=150)


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0)

print("TEST ACCURACY",test_acc)

plt.plot(history.history['loss'], label='train_loss',c='blue')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='Training acc',c='blue')
plt.plot(val_acc,label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


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
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'

# %%
# Try different numbers of n_estimators - this will take a minute or so
n_neighbors = np.arange(1, 50, 1)
scores = []
for n in n_neighbors:
    model.set_params(n_neighbors=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Score")
plt.plot(n_neighbors, scores)

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
