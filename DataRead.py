
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
# dirpath = os.getcwd() 
# print("current directory is : " + dirpath)
# path2add = ''
filepath1 ='C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore_user_reviews.csv'
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


#%%

#View entire data set 
pdata.head()
# pdata.describe
pdata.shape
#%%
#View User Review Data
userdata.head()
userdata.shape

# %%
# Establish standard quick checks functions:
def dfChkBasics(dframe): 
  cnt = 1
  
  try:
    print(str(cnt)+': info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(str(cnt)+': describe(): ')
  cnt+=1
  print(dframe.describe())

  print(str(cnt)+': dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(str(cnt)+': columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(str(cnt)+': head() -- ')
  cnt+=1
  print(dframe.head())

  print(str(cnt)+': shape: ')
  cnt+=1
  print(dframe.shape)

  # print(str(cnt)+': columns.value_counts(): ')
  # cnt+=1
  # print(dframe.columns.value_counts())

def dfChkValueCnts(dframe):
  cnt = 1
  for i in dframe.columns :
      print(str(cnt)+':', i, 'value_counts(): ')
      print(dframe[i].value_counts())
      cnt +=1 


#%%
# Run standard checks on the google playstore data frame and google playstore reviews data frame
dfChkBasics(pdata)
dfChkValueCnts(pdata)

dfChkBasics(userdata)
dfChkValueCnts(userdata)

#%%
