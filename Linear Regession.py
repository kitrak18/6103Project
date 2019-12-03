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
# Correlation matrix
cor_matrix = pdata_clean[['Rating', 'Reviews', 'Installs', 'Price']].corr()
f, ax = plt.subplots()
p =sns.heatmap(cor_matrix, annot=True, cmap="YlGnBu")

#%%[markdown]
# Linear Regression
# Rating ~ Reviews + Installs + Price
from statsmodels.formula.api import ols
modelGreGpa = ols(formula='Rating ~ Reviews + Installs + Price', data=pdata_clean).fit()
print( modelGreGpa.summary() )

# R^2 and Adjusted R^2 are very low (0.004 and 0.003, respectively).
# Only Reviews has a significant p-value, which means Installs and Price don't have a significant effect on Ratings.

#%%[markdown]
# Linear Regression
# Rating ~ C(Category) + Reviews + Installs + Price + C(Type)
modelGreGpa = ols(formula='Rating ~ C(Category) + Reviews + Installs + Price + C(Type)', data=pdata_clean).fit()
print( modelGreGpa.summary() )

# R^2 is improved, but still low, 0.032. Adjusted R^2 is 0.028.