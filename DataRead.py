
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
# Convert Reviews, Installs, and Price variables into numeric
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
fig = sns.barplot(x=pdata_clean['Category'], y=pdata_clean['Rating'], palette="hls")
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

#%%[markdown]
# Plot Rating, Reviews, Installs, and Price matrix
## Scaled data
from sklearn.preprocessing import scale
df = pdata_clean[['Rating', 'Reviews', 'Installs', 'Price']]
dfscale = pd.DataFrame(scale(df), columns=df.columns)
sns.set()
sns.pairplot(dfscale)

## Take log of Reviews and Installs
df['Log Reviews'] = np.log(df.Reviews)
df['Log Installs'] = np.log(df.Installs)
sns.pairplot(df)

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

