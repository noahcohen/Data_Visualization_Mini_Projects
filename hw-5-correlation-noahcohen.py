
# coding: utf-8

# # Homework 5 (Partial) Solution
# 
# Notes:
# - __This is NOT a sample homework submission.__ The purpose of this notebook is for you to verify the correctness of your results. We will post selected excellent homework submissions separately. 
# - The solutions provided here are for reference only. It is very likely that more than one solutions exist for a problem.
# - If you think there is any error in this notebook, please email `si370-staff@umich.edu`.

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns


# In[15]:

# To silent warnings caused by a recent (Oct 2015) matplotlib update
import warnings
warnings.filterwarnings('ignore')


# In[16]:

# Load and clean data
df = pd.read_csv('movies-2001-2010.csv')
rdate = pd.to_datetime(df['ReleaseDate'])
df['ReleaseDate'] = rdate
df.shape


# In[17]:

#2(a-1)
df_drama  = df[df.Genre=='Drama']
df_drama.shape


# In[18]:

#2(a-2)
df_drama.ix[df_drama.Budget.idxmax()]


# In[19]:

#2(a-3)
df_drama.ix[df_drama.Gross.idxmax()]


# In[20]:

#2(b)
df_drama[['Budget']].plot(kind='box',
                          title='Budget of Drama Films');


# In[21]:

# 2(c)
sns.boxplot(x='CreativeType', y='Budget', data=df_drama);
plt.title('Budget of Drama Films by Creative Type');


# In[22]:

# 2(d)
plt.figure(figsize=(6,6));
df_drama.groupby('CreativeType').size().plot(kind='pie');
plt.title('Creative Type Distribution by #Films Among Dramas');
plt.ylabel('');


# In[23]:

# 2(e)
df_drama = df_drama.copy()
df_drama['ProfitMargin'] = (df_drama['Gross'] - df_drama['Budget'])    / df_drama['Gross']
sns.boxplot(x='CreativeType', y='ProfitMargin', data=df_drama);
plt.ylim(-5,1)


# In[24]:

# 2(f)
df_drama_clean = df_drama[df_drama.ProfitMargin>=-1]
sns.pairplot(df_drama_clean[['Budget','Gross','ProfitMargin',
                             'RunningTime','RottenTomatoesRating']]);


# In[25]:

# 2(g)
# Show a bar plot of correlations
methods = ['pearson','spearman','kendall']
fields = ['Budget', 'Gross', 'ProfitMargin']
cols = {}
for m in methods:
    col = [df_drama_clean['RunningTime'].corr(df_drama_clean[f], method=m)
           for f in fields]
    cols[m] = col
df_corrs_with_running_time = pd.DataFrame(cols, columns=methods, 
                                          index=fields)
df_corrs_with_running_time.plot(kind='barh',
                                title='Correlation with RunningTime');


# In[26]:

# 2(h)
sns.lmplot(x='RunningTime', y='Budget', data=df_drama_clean,
           truncate=True);


# In[27]:

model1 = smf.ols('Budget ~ RunningTime', data=df_drama_clean).fit()
model1.summary()

