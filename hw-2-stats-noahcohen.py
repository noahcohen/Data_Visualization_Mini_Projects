
# coding: utf-8

# # Homework 2 Solution
# 
# Notes:
# - __This is NOT a sample homework submission.__ The purpose of this notebook is for you to verify the correctness of your results. We will post selected excellent homework submissions separately. 
# - The solutions provided here are for reference only. It is very likely that more than one solutions exist for a problem.
# - If you think there is any error in this notebook, please email `si370-staff@umich.edu`.
# 

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().magic(u'matplotlib inline')


# In[2]:

# Import Data
df_raw = pd.read_csv('nutritions.csv')
df_raw.shape


# In[3]:

# 2(a) Remove all rows that have any invalid values (NA).
df_a = df_raw.dropna()
df_a.shape


# In[4]:

# 2(b) Remove the food entries whose five 
#      basic proximates do not sum to to 100g, 
#      with 1g tolerance on both sides.
proximates = ['Ash', 'Fat', 'Protein', 'Water', 'Carbohydrate']
df_b = df_a[np.isclose(df_a[proximates].sum(axis=1), 100, atol=1.0, rtol=0)]
df_b.shape


# In[5]:

# 2(c) Report the number of entries that are removed at each step.
print('%d entries removed in (a)'%(len(df_raw) - len(df_a)))
print('%d entries removed in (b)'%(len(df_a) - len(df_b)))


# In[6]:

# 3(a) What is the mean Carbohydrates for all food entries?
df = df_b
df.Carbohydrate.mean()


# In[7]:

# 3(b) What is the mean Proteins for each group?
df.groupby('Group')[['Protein']].mean()


# In[8]:

# 3(c) Which food has the highest Proteins for each group? 
by_group = df.groupby('Group')
by_group.apply(lambda x: df.ix[x['Protein'].idxmax(), ['Food','Protein']])


# In[9]:

# 3(d) Which group has the highest mean Energy?
energy_by_group = by_group[['Energy']].mean()
energy_by_group.ix[energy_by_group.idxmax()]


# In[10]:

# 3(e) Create a histogram showing the distribution of Fat for Fast Foods.
df.ix[df.Group=='Fast Foods', 'Fat'].hist();
plt.title('Distribution of Fat among Fast Foods');
plt.xlabel('Fat g/100g');
plt.ylabel('Count');


# In[11]:

# 3(f) Create a boxplot to compare the distributions of all nutritions 
#      for Fast Foods.
df.ix[df.Group=='Fast Foods', proximates].boxplot(return_type='axes');
plt.ylabel('Weight g / 100g');
plt.title('Distribution of Proximates among Fast Foods');


# In[12]:

# 3(g) reate a boxplot to compare the distribution of Energy for 
#      Fast Foods vs. the distribution of Energy for Sweets.
energy_ff = df.ix[df.Group=='Fast Foods', 'Energy']
energy_sw = df.ix[df.Group=='Sweets', 'Energy']
energy_ffsw = pd.concat((energy_ff,energy_sw), axis=1)
energy_ffsw.columns = ['Fast Foods', 'Sweets']
energy_ffsw.boxplot(return_type='axes');
plt.title('Distribution of Energy');


# In[13]:

# 3(h) Perform a two-sided t-test for the null hypothesis 
#      that the Energy for Fast Foods and the Energy for Sweets 
#      have identical average values. Explain what the result means.
import scipy.stats
scipy.stats.ttest_ind(energy_ff, energy_sw, equal_var=False)


# Using $\alpha=0.05$ as the threshold, p-value is smaller than the threshold. We reject the null hypothesis. Therefore, Fast Foods and Sweets do not have identical average values.

# In[14]:

# 4(a) What is the Pearson's correlation between Energy and 
#      Carbohydrate among all food entries?
df.Energy.corr(df.Carbohydrate)


# In[15]:

# 4(b) Generate the correlation matrix between Energy, Carbohydrate, 
#      Fat, Protein, Ash, and Water.
nutritions = ['Energy','Carbohydrate','Fat','Protein','Ash','Water']
cormat = df[nutritions].corr()
cormat


# In[16]:

# 4(c) Visualize the correlation matrix.
fig, ax = plt.subplots()
cax = ax.matshow(cormat, cmap=plt.cm.OrRd)
fig.colorbar(cax);

tags = [''] + cormat.columns.tolist()
tags[2] = 'Carbs'
ax.set_xticklabels(tags);
ax.set_yticklabels(tags);
plt.title('Correlation Among Nutrients', y=-0.1);
plt.grid(False);


# In[17]:

# 4(d) Build a simple regression model of Energy from Fat, using 
#      entries in Fast Foods only. This means Energy is the dependent 
#      variable (Y), and Fat is the independent variable (X). Report 
#      coefficients, R-squared score, and the confidence intervals of 
#      the coefficients.
model1 = smf.ols('Energy ~ Fat', data=df[df.Group=='Fast Foods']).fit()
model1.summary()


# Results:
# - Coefficients:
#   - Intercept: 134.6526
#   - Fat: 9.5708
# - R-squared score : 0.700
# - 95% Confidence Intervals:
#   - Intercept: [122.963, 146.342]
#   - Fat: [8.774, 10.368]
