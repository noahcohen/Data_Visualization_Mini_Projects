
# coding: utf-8

# # Homework 1 Solution
# 
# Notes:
# - __This is NOT a sample homework submission.__ The purpose of this notebook is for you to verify the correctness of your results. We will post selected excellent submissions by students separately. 
# - The solutions provided here are for reference only. It is very likely that more than one solutions exist for a problem.
# - If you think there is any error in this notebook, please email `si370-staff@umich.edu`.
# 

# In[1]:

import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')


# In[2]:

df_1995 = pd.read_csv('names/yob1995.txt')
dfs = []
years = range(1880, 2015)
for y in years:
    df = pd.read_csv('names/yob%d.txt' % y)
    df['year'] = y
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)


# In[3]:

# 4(a) What are the 10 most popular names in 1995?
df_1995.sort('birth_count', ascending=False).head(10)


# In[4]:

# 4(b) What are the 10 most popular boys' names in 1995?
df_1995.query('gender=="M"').sort('birth_count', ascending=False).head(10)


# In[5]:

# 4(c) What are the 10 most popular girls' names in 1995?
df_1995.query('gender=="F"').sort('birth_count', ascending=False).head(10)


# In[6]:

# 4(d) What is the average birth_count of all names in 1995?
df_1995.birth_count.mean()


# In[7]:

# 4(e) What is the median birth_count of boys' names in 1995?
df_1995.query('gender=="M"').birth_count.median()


# In[8]:

# 4(f) What is the maximum birth_count of girls' names in 1995?
df_1995.query('gender=="F"').birth_count.max()


# In[9]:

# 5(a) What is the total #births for each year?
total_births_by_year = df_all.groupby('year')[['birth_count']].sum()
total_births_by_year.head()


# In[10]:

# 5(b) What is the total #births for each year for each gender?
total_births_by_year_gender = df_all.groupby(
    ('year','gender'))[['birth_count']].sum()
total_m_f = total_births_by_year_gender.unstack()['birth_count'][['M','F']]
total_m_f.head()


# In[11]:

# 5(c) Which year has the maximum total #births?
total_births_by_year.ix[total_births_by_year.idxmax()]


# In[12]:

# 5(d) Which year has the maximum total #births for boys?
total_m_f.ix[[total_m_f.M.idxmax()],['M']]


# In[13]:

# 5(e) Which year has the biggest difference between #boys and #girls 
#     (i.e., abs(#boys-#girls))?
total_m_f.ix[[np.argmax(np.abs(total_m_f.M - total_m_f.F))]]


# In[14]:

# 5(f) What are the 5 most popular boys' names in the 1990s (1990-1999)?
df_m1990s = df_all.query('gender=="M" and year >= 1990 and year <= 1999')
df_m1990s.groupby('name')[['birth_count']]     .sum().sort('birth_count', ascending=False).head()


# In[15]:

# 5(g) What are the 5 most popular girls' names 
#      in the 20th century (1901-2000)?
df_g20c = df_all.query('gender=="F" and year>=1901 and year<=2000')
df_g20c.groupby('name')[['birth_count']]     .sum().sort('birth_count', ascending=False).head()


# In[16]:

# 5(h) What are the 15 names with the highest 
#      gender neutrality in the 1990s?
#      Define the gender neutrality of a name as min(#boys, #girls).
df_1990s = df_all.query('year>=1990 and year<=1999')
df_mf1990s = df_1990s.groupby(('name','gender'))     .sum().unstack()['birth_count'].fillna(0)
df_mf1990s['neutrality'] = df_mf1990s.min(axis=1)
df_mf1990s.sort('neutrality', ascending=False).head(15)


# In[17]:

# 6(a) Plot the total #births by gender and year, as below.
total_m_f.plot(title='Total births by gender and year');


# In[19]:

# 6(b) Plot the proportion of the name "Mike" among all male 
#      names by year, as below.
mike = df_all.query('name=="Mike" and gender=="M"').copy()
mike = mike.set_index('year')
mike['total_male'] = total_m_f.M
mike['proportion'] = mike.birth_count / mike.total_male
mike.plot(y='proportion', title='Proportion of "Mike" among boy names');

