
#%% 
# ## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
import math

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

pdatac = pdata.drop_duplicates(subset=['App'], keep = 'last')
print(pdatac.shape)
print(pdatac.info())

# Keep the last instance because it is the most recent version of the App

#%%[markdown]
# There are NA values in Rating, Type, Content Rating, Current Version, and Android Version
NanValues = pd.DataFrame(pdata, columns = pdata.columns)
NanChart = NanValues.isnull().sum()
NanChart

# Chart the number of NaN values for each variable 
fig = sns.barplot(x=NanChart.index,y=NanChart.values, palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Counts of NaN Values')

for i in fig.patches:
  fig.text(i.get_x(), i.get_height() + 20, \
      str(round((i.get_height()), 2)), fontsize=11, color='black', rotation=0)

plt.show(fig)

#%%
# Let's clean out these NaN values
pdata_clean = pdatac.dropna()
pdata_clean.info()

print(pdata_clean.info())
# We have 8190 rows now.

#%%[markdown]

print(pdata_clean.Category.value_counts())

# Plot count of apps for each Category
plt.figure(figsize=(10, 7))
category = pdata_clean["Category"].value_counts()[:34]
fig = sns.barplot(x=category.values, y=category.index, palette="viridis")
plt.title('Categories and their counts')
plt.show(fig)

#%%[markdown]
# Histogram for Rating
pdata_clean.Rating.hist()
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')

#%%[markdown]
# Bar chart for Catergory and Rating
plt.figure(figsize=(8,6))
fig = sns.barplot(x=pdata_clean['Category'], y=pdata_clean['Rating'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Category and Rating')
plt.show(fig)

#%%[markdown]
print(pdata_clean['Type'].value_counts())

# Plot counts for each app type (free vs. paid apps)
fig = sns.countplot(x=pdata_clean['Type'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Counts for paid and free apps')
plt.show(fig)
# There are significantly more free apps on the GooglePlay Store than paid apps.


# Boxplot for Type and Rating
plt.figure(figsize=(8,6))
fig = sns.boxplot(x=pdata_clean['Type'], y=pdata_clean['Rating'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Type and Rating')
plt.show(fig)
# Comparing the distribution of ratings between paid and free apps shows that paid apps have a higher mean rating than free apps. This would be an interesting relationship to potentially explore.

#%%

# Is there a relationship between the size of an app and the rating? I will use a side by side axis chart to visualize the two variables.

# First I will clean up the Size variable, which is currently a factor. SizeNum will be the Size variable converted to numeric.
pdata_clean['SizeNum']=pdata_clean['Size'].fillna(0)
pdata_clean['SizeNum'] = pdata_clean['SizeNum'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
pdata_clean['SizeNum'] = pdata_clean['SizeNum'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
pdata_clean['SizeNum']=pdata_clean['SizeNum'].apply(lambda x:str(x).replace(',','')if 'M' in str(x) else x)
pdata_clean['SizeNum']=pdata_clean['SizeNum'].apply(lambda x:float(str(x).replace('k',''))/1000 if 'k'in str(x) else x)
pdata_clean['SizeNum']=pdata_clean['SizeNum'].apply(lambda x:float(x))

# Plot the size by side axis chart
sns.jointplot(pdata_clean['SizeNum'],pdata_clean['Rating'],kind='kde',color='lightblue')

# There appears to be an interesting relationship between size of the application and rating. Smaller apps have ratings concentrated around 4.0 - 4.5 and the size of apps displays a slightly normal distribution (although right skewed) and ranges from 0M - 60M.

#%%

# Is there a relationship between rating and price? For paid apps, we will visualize the relationship with price.

pdata_clean['PriceNum'] = pdata_clean['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
pdata_clean['PriceNum'] = pdata_clean['PriceNum'].apply(lambda x: float(x))

paid_apps = pdata_clean[pdata_clean.PriceNum > 0]

sns.jointplot( "PriceNum", "Rating", paid_apps,color='purple', marginal_kws=dict(bins = 25, rug = True), annot_kws = dict(stat = "r"), kind = 'scatter')

# It looks like there are some far outliers. Let's take a look at what they are
pdata_clean[['Category','App','PriceNum','Rating']][pdata_clean.PriceNum > 70]
# Most seem to be a variation of "I Am Rich" and fall under the family, lifestyle, and finance categories. They are all $399.99 except for the $400 "Trump Edition".

# I will remove the "rich" outliers and re-chart the data
paid_apps2 = paid_apps[paid_apps.PriceNum < 70]

sns.jointplot( "PriceNum", "Rating", paid_apps2,color='purple', marginal_kws=dict(bins = 25, rug = True), annot_kws = dict(stat = "r"), kind = 'hex')

# It's definitely easier to see the relationship between price and rating now with the outliers removed and heat chart applied. It seems that most highly rated apps are in the lower price range, but most apps overall are in that same range. There does not seem to be a clear correlation here. Correlations will be explored in the next chunk.


#%%

# Correlation plot between numeric variables
# First, I will convert installs and reviews to numeric variables

pdata_clean['InstallsNum'] = pdata_clean['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
pdata_clean['InstallsNum'] = pdata_clean['InstallsNum'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
pdata_clean['InstallsNum']=pdata_clean['InstallsNum'].apply(lambda x:float(x))

pdata_clean['ReviewsNum'] = pdata_clean['Reviews'].apply(lambda x: int(x))

# Plot the correlation of numeric variables
correlation = pdata_clean.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm_r")
# This plot shows that not many of the variables are strongly correlated, except for installs and reviews. This intrinsically makes sense because the larger number of installs an app has, the more likely it is to be reviewed.

#%%[markdown]
print(pdata_clean['Installs'].value_counts())

# Aggregate the counts by Installs
plt.figure(figsize=(20,5))
fig = sns.countplot(x=pdata_clean['Installs'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Number of installs and their counts')
plt.show(fig)

# Bar chart for Installs and Rating
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
fig2 = sns.countplot(y=pdata_clean['Genres'], palette="hls")
fig2.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.title('Genres and their counts')
plt.show(fig)

# We'll narrow it down to show the top 20 to get a sense of the most dominant genres in the Google Play Store

plt.figure(figsize=(10, 7))
genres = pdata_clean["Genres"].value_counts()[:20]
fig3 = sns.barplot(x=genres.values, y=genres.index, palette="GnBu_r")
plt.title('Top 20 Genres and their counts')
# fig3.savefig('C:/Users/sjg27/OneDrive/Documents/GWU Data Science/Fall 19/DATS 6103 Intro to DM/Project/genreplot.png')

# The top three genres with the most apps in the Google Play Store are Tools, Entertainment, and Education

#%%[markdown]
# Plot Rating, Reviews, Installs, Price, and Size matrix
## Scaled data
from sklearn.preprocessing import scale
df = pdata_clean[['Rating', 'ReviewsNum', 'InstallsNum', 'PriceNum','SizeNum']]
dfscale = pd.DataFrame(scale(df), columns=df.columns)
sns.set()
sns.pairplot(dfscale)

## Take log of Reviews and Installs
df['Log ReviewsNum'] = np.log(df.ReviewsNum)
df['Log InstallsNum'] = np.log(df.InstallsNum)
fig = sns.pairplot(df)
# fig.savefig('C:/Users/sjg27/OneDrive/Documents/GWU Data Science/Fall 19/DATS 6103 Intro to DM/Project/pairplot.png')

#%%[markdown] 
# Focus on the scatterplot of Rating vs Reviews
plt.figure(figsize=(8,6))
fig = sns.scatterplot(x=pdata_clean['ReviewsNum'], y=pdata_clean['Rating'], palette="hls")
#fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.title('Rating vs Reviews')
plt.show(fig)

#%%[markdown]
# # K-means 
from sklearn.cluster import KMeans

xdata = pdata_clean[['Rating', 'Reviews', 'Installs', 'Price']]

km_xdata = KMeans( n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xdata.fit_predict(xdata)

# plot the 3 clusters
index1 = 0
index2 = 1

plt.scatter( xdata[y_km==0].iloc[:,index1], xdata[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xdata[y_km==1].iloc[:,index1], xdata[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xdata[y_km==2].iloc[:,index1], xdata[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

# plot the centroids
plt.scatter( km_xdata.cluster_centers_[:, index1], km_xdata.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
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
# DESCRIBE CHARTS
groups = pdata_clean.groupby('Category').filter(lambda x: len(x) > 286).reset_index()
array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))

