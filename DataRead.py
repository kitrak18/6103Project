
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
filepath1 ='C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore.csv'
pdata = pd.read_csv(filepath1)

filepath2 = 'C:/Users/karti/OneDrive/Documents/6103Project/googleplaystore_user_reviews.csv'
userdata = pd.read_csv(filepath2)

#%% [markdown]
#Trinh
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