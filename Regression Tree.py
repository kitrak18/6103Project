
#%% 
# ## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')

#%% 
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
#Trinh test commmit
import os
import pandas as pd
dirpath = os.getcwd() 
#print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='C:\\Users\\Trinh\\Desktop\\GWU\\Fall 2019\\Intro to Data Mining\\Project\\googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'C:\\Users\\Trinh\\Desktop\\GWU\\Fall 2019\\Intro to Data Mining\\Project\\googleplaystore_user_reviews.csv'
userdata = pd.read_csv(filepath2)

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

#%%[markdown]
# The dataset has 10841 rows and 13 columns. There are duplicates in App names. Let's drop duplicates.

#%%
pdatac = pdata.drop_duplicates(subset=['App'], keep = 'first')
print(pdatac.shape)
print(pdatac.info())

#%%[markdown]
# There are NA values in Rating, Type, and Content Rating. Let's remove them

pdata_clean = pdatac.dropna()
pdata_clean.info()

# We have 8190 rows now.

#%%[markdown]
# Convert Reviews, Installs, Price , and Size variables into numeric
# Start with Reviews
print(pdata_clean.Reviews.value_counts())
pdata_clean.Reviews = pd.to_numeric(pdata_clean.Reviews, errors='coerce')

# Installs
print(pdata_clean.Installs.value_counts())
#Let's remove '+' and ','
pdata_clean.Installs = pdata_clean.Installs.apply(lambda x: x.strip('+'))
pdata_clean.Installs = pdata_clean.Installs.apply(lambda x: x.replace(',',''))

pdata_clean.Installs = pd.to_numeric(pdata_clean.Installs, errors='coerce')

# Price
print(pdata_clean.Price.value_counts())
pdata_clean.Price = pdata_clean.Price.apply(lambda x: x.strip('$'))
pdata_clean.Price = pd.to_numeric(pdata_clean.Price, errors='coerce')
#%%
# Size
pdata_clean.Size = pdata_clean.Size.apply(lambda x: x.strip('M'))
pdata_clean.Size = pdata_clean.Size.apply(lambda x: x.strip('k'))
pdata_clean.Size = pd.to_numeric(pdata_clean.Size, errors='coerce')

#%%[markdown]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(pdata_clean.Category.value_counts())

# Plot counts for each Category
plt.figure(figsize=(20,5))
fig = sns.countplot(x=pdata_clean['Category'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Categories and their counts')
plt.show(fig)

#%%[markdown]
# Histogram for Rating
pdata_clean.Rating.hist()
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')

#%%[markdown]
# Boxplot for Catergory and Rating
plt.figure(figsize=(8,6))
fig = sns.boxplot(x=pdata_clean['Category'], y=pdata_clean['Rating'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Category and Rating')
plt.show(fig)

#%%[markdown]
print(pdata_clean['Type'].value_counts())

# Plot counts for each Type (free app and paid app)
fig = sns.countplot(x=pdata_clean['Type'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Counts for paid and free apps')
plt.show(fig)

# Boxplot for Type and Rating
plt.figure(figsize=(8,6))
fig = sns.boxplot(x=pdata_clean['Type'], y=pdata_clean['Rating'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Type and Rating')
plt.show(fig)

#%%[markdown]
print(pdata_clean['Installs'].value_counts())

# Plot counts for Installs
plt.figure(figsize=(20,5))
fig = sns.countplot(x=pdata_clean['Installs'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Number of installs and their counts')
plt.show(fig)

# Boxplot for Installs and Rating
plt.figure(figsize=(8,6))
fig = sns.boxplot(x=pdata_clean['Installs'], y=pdata_clean['Rating'], palette="hls", )
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Installs and Rating')
plt.show(fig)

# Boxplot for Installs and Reviews
plt.figure(figsize=(8,6))
fig = sns.boxplot(x=pdata_clean['Installs'], y=pdata_clean['Reviews'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Installs and Reviews')
plt.show(fig)

#%%[markdown] 
# Scatterplot Rating vs Reviews
plt.figure(figsize=(8,6))
fig = sns.scatterplot(x=pdata_clean['Reviews'], y=pdata_clean['Rating'], palette="hls")
#fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Rating vs Reviews')
plt.show(fig)

#%%[markdown]
print(pdata_clean['Content Rating'].value_counts())

# Plot counts for Content Rating
fig = sns.countplot(x=pdata_clean['Content Rating'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Content Rating and their counts')
plt.show(fig)

#%%[markdown]
print(pdata_clean.Genres.value_counts())

# Plot counts for Genres
plt.figure(figsize=(15,20))
fig = sns.countplot(y=pdata_clean['Genres'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Genres and their counts')
plt.show(fig)

#%%
pdata_clean['SizeNum']=pdata_clean['Size'].dropna()
pdata_clean['SizeNum'] = pdata_clean['SizeNum'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
pdata_clean['SizeNum'] = pdata_clean['SizeNum'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
pdata_clean['SizeNum']=pdata_clean['SizeNum'].apply(lambda x:str(x).replace(',','')if 'M' in str(x) else x)
pdata_clean['SizeNum']=pdata_clean['SizeNum'].apply(lambda x:float(str(x).replace('k',''))/1000 if 'k'in str(x) else x)
pdata_clean['SizeNum']=pdata_clean['SizeNum'].apply(lambda x:float(x))

#%%[markdown]
# Plot Rating, Reviews, Installs, and Price matrix
## Scaled data
from sklearn.preprocessing import scale
df = pdata_clean[['Rating', 'Reviews', 'Installs', 'Price', 'SizeNum']]
dfscale = pd.DataFrame(scale(df), columns=df.columns)
sns.set()
sns.pairplot(dfscale)

## Take log of Reviews and Installs
df['Log Reviews'] = np.log(df.Reviews)
df['Log Installs'] = np.log(df.Installs)
sns.pairplot(df)

#%%[markdown]
# Correlation matrix
cor_matrix = pdata_clean[['Rating', 'Reviews', 'Installs', 'Price']].corr()
f, ax = plt.subplots()
p =sns.heatmap(cor_matrix, annot=True, cmap="YlGnBu")

#%%
#sns.regplot('Rating', 'Reviews', data = dfscale, x_jitter = 5,fit_reg = True, line_kws = {'color':'red', 'label':'LM fit'})

#%%[markdown]
# Linear Regression
# Rating ~ Reviews + Installs + Price
from statsmodels.formula.api import ols
model1 = ols(formula='Rating ~ Reviews + Installs + Price + Size', data=pdata_clean).fit()
print( model1.summary() )

# R^2 and Adjusted R^2 are very low (0.006 and 0.005, respectively).
# Only Reviews has a significant p-value, which means Installs and Price don't have a significant effect on Ratings.

#%%[markdown]
# Linear Regression
# Rating ~ Reviews + Installs + Price
model2 = ols(formula='Rating ~ Reviews + Installs + Price + SizeNum + C(Type)', data=pdata_clean).fit()
print( model2.summary() )

#%%
from sklearn import linear_model
fit1 = linear_model.LinearRegression()  # instantiate the object, with full functionality

xdata = pdata_clean[['Reviews', 'Installs', 'Price', 'Size']]
ydata = pdata_clean[['Rating']]

fit1.fit( xdata , ydata )
fit1.score( xdata, ydata)


#%%[markdown]
# K-means 
from sklearn.cluster import KMeans

xdata = pdata_clean[['Rating', 'Reviews', 'Installs', 'Price']]

km_xdata = KMeans( n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 3 clusters
index1 = 1 #Reviews
index2 = 0 #Rating

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(loc = 'bottom right', scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# plot the 3 clusters
index1 = 2 #Installs
index2 = 0 #Rating

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(loc = 'bottom right', scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# plot the 3 clusters
index1 = 3 #Price
index2 = 0 #Rating

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(loc = 'bottom right', scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
km_xdata = KMeans( n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 4 clusters
index1 = 1 #Reviews
index2 = 0 #Rating

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

plt.scatter( xdata[y_km==3].iloc[:,index1], xdata[y_km==3].iloc[:,index2], s=50, c='pink', marker='p', edgecolor='black', label='cluster 4' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(loc = 'bottom right', scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()
#%%
# 2 clusters
# Rating and Reviews
plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# 2 clusters
# Reviews and Installs
index1 = 1
index2 = 2

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# 2 clusters
# Installs and Price
index1 = 2
index2 = 3

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# 2 clusters
# Rating and Installs
index1 = 0
index2 = 2

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# 2 clusters
# Rating and Price
index1 = 0
index2 = 3

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()
#%%
# 4 clusters
# Rating and Installs
index1 = 0
index2 = 2

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

plt.scatter( xdata[y_km==3].iloc[:,index1], xdata[y_km==3].iloc[:,index2], s=50, c='purple', marker='p', edgecolor='black', label='cluster 4' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# 4 clusters
# Rating and Reviews
index1 = 0
index2 = 1

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

plt.scatter( xdata[y_km==3].iloc[:,index1], xdata[y_km==3].iloc[:,index2], s=50, c='purple', marker='p', edgecolor='black', label='cluster 4' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(xdata.columns[index1])
plt.ylabel(xdata.columns[index2])
plt.grid()
plt.show()

#%%
# Decision Tree
xdata = pdata_clean[['Reviews', 'Installs', 'Price']]
ydata = pdata_clean[['Rating']]

# Regression Trees

from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error as MSE  
# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(xdata, ydata, test_size=0.2,random_state=1)
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree0 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1,random_state=22) # set minimum leaf to contain at least 10% of data points
# DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
#     max_leaf_nodes=None, min_impurity_decrease=0.0,
#     min_impurity_split=None, min_samples_leaf=0.13,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=3, splitter='best')

regtree0.fit(X_train, y_train)  # Fit regtree0 to the training set

# evaluation
y_pred = regtree0.predict(X_test)  # Compute y_pred
mse_regtree0 = MSE(y_test, y_pred)  # Compute mse_regtree0
rmse_regtree0 = mse_regtree0 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0))

#%%
# Let us compare the performance with OLS
from sklearn import linear_model
olsdata = linear_model.LinearRegression() 
olsdata.fit( X_train, y_train )

y_pred_ols = olsdata.predict(X_test)  # Predict test set labels/values

mse_ols = MSE(y_test, y_pred_ols)  # Compute mse_ols
rmse_ols = mse_ols**(0.5)  # Compute rmse_ols

print('score:', olsdata.score(X_test, y_test)) # 0.9980384387631105
print('intercept:', olsdata.intercept_) # 5.4722114861507745
print('coef_:', olsdata.coef_) 

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree1 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=0.22, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
from sklearn.model_selection import cross_val_score
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(regtree1, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree1.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree1.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree1.predict(X_test)  # Predict the labels of test set

print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(y_test, y_predict_test)**(0.5) )

#%%
# Plot the tree
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
import os
dirpath = os.getcwd() # 
#print("current directory is : " + dirpath)
# path2add = 'GWU_classes/DATS_6103_DataMining/Class11_Trees_SVM'
filepath = os.path.join( dirpath, 'tree1')
export_graphviz(regtree1, out_file = filepath+'.dot' , feature_names =['Reviews', 'Installs', 'Price']) 


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


#%%

#View entire data set 
pdata.head()
# pdata.describe
pdata.shape
#%%
#View User Review Data
userdata.head()
userdata.shape


#%%

# Histogram of rating by category
groups = pdata.groupby('Category').filter(lambda x: len(x) > 286).reset_index()
array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))

