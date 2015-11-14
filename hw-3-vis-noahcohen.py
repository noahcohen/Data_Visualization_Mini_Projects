
# coding: utf-8

# # Homework 3 Solution
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
get_ipython().magic(u'matplotlib inline')
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
sns.set(color_codes=True)


# In[2]:

# 1. Import and clean data
df = pd.read_csv('nutrients/cleaned-for-student/nutritions.csv')

# Cleaning: Step-1
df = df.dropna()

# Cleaning: Step-2
proximates = ['Protein','Fat','Carbohydrate','Ash','Water']
total_proximates = df[proximates].sum(1)
df = df[np.isclose(total_proximates, 100, rtol=0, atol=1.0)]

df.shape


# In[3]:

# 2a. Create a scatter plot to display the values of Energy and 
#     Fat for Fast Foods and Sweets.
sns.lmplot(x='Fat',y='Energy',hue='Group',fit_reg=False,
           data=df[df.Group.isin(('Fast Foods', 'Sweets'))]);


# In[4]:

# 2b. Create a bar plot to compare the mean Fat value of food items 
#     that contain "egg", "apple", and "chocolate" respectively.
df['Ingridient'] = ''
ingridients = ['Egg', 'Apple', 'Chocolate']
for ingr in ingridients:
    ingr_idx = df.Food.str.lower().str.contains(ingr.lower())
    df.ix[ingr_idx, 'Ingridient'] = ingr
sns.barplot(x='Ingridient', y='Fat', hue='Group',
            data=df[df.Group.isin(('Sweets','Fast Foods')) 
                    & df['Ingridient']]);


# In[5]:

# 2c. Create a box plot for the same groups of data as above.
sns.boxplot(x='Ingridient', y='Fat', hue='Group',
            data=df[df.Group.isin(('Sweets','Fast Foods')) 
                    & df['Ingridient']]);


# In[6]:

# 2d. Create a stacked bar chart comparing the mean values of all 
#     five proximates (fat, protein, carbs, ash, and water) for 
#     all food groups.
df.groupby('Group')[proximates].mean().plot(kind='barh', stacked=True);
plt.legend(bbox_to_anchor=(1.3,1));
plt.xlabel('g / 100g');
plt.xlim(0,100);


# In[7]:

# 2e. Create two histograms showing the distribution of Fat for 
#     Fat Foods, using 10 bins and 100 bins respectively.
fig, axes = plt.subplots(ncols=2, figsize=(12,5))
fast_foods_fat = df[df.Group=='Fast Foods'].Fat
sns.distplot(fast_foods_fat, bins=10, ax=axes.flat[0], kde=False);
sns.distplot(fast_foods_fat, bins=100, ax=axes.flat[1], kde=False);
axes[0].set_xlabel('Fat g / 100g');
axes[1].set_xlabel('Fat g / 100g');
axes[0].set_title('bins=10');
axes[1].set_title('bins=100');
fig.suptitle('Fat distribution among Fast Foods', fontsize=14);


# In[8]:

# 2f. Create a scatter plot matrix (SPLOM) displaying the 
#     relations among Energy, Fat, Protein, and Carbohydrate.
sns.pairplot(df[df.Group.isin(('Sweets','Fast Foods'))], 
             vars = ['Energy', 'Fat','Protein','Carbohydrate'],
             hue='Group');


# In[9]:

# 2g. Create two scatter plots with a fitted curve displaying 
#     the relationship of Energy vs. Fat among Fast Foods. Use 
#     linear fit and 2nd-order polynomial fit respectively.
df_ff = df[df.Group == 'Fast Foods']
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.regplot(x='Fat',y='Energy',data=df_ff, ax=axes[0]);
sns.regplot(x='Fat',y='Energy',order=2,data=df_ff, ax=axes[1]);
axes[0].set_title('Order=1 (Linear)');
axes[1].set_title('Order=2');
fig.suptitle('Energy vs. Fat among Fast Foods', fontsize=14);


# In[10]:

# 2h. Create a scatterplot with marginal histograms for 
#     Energy vs. Fat among Fast Foods.
sns.jointplot(x='Fat',y='Energy',data=df_ff);


# In[11]:

# 3a. Create a linear regression model of Energy from Fat for 
#     Fast Foods, which should be identical to the one created 
#     in the previous homework.
df_ff = df[df.Group == 'Fast Foods']
model1 = smf.ols('Energy ~ Fat', data=df_ff).fit()
model1.summary()


# In[12]:

# 3b. Which item in Fast Foods has the highest Fat? Do you 
#     think (make a guess) if it is an outlier in the regression model?
food_maxfat = df_ff.ix[[df_ff.Fat.idxmax()], ['Food','Fat','Energy']]
food_maxfat


# In[13]:

sns.lmplot(x='Fat',y='Energy',data=df_ff);

# Annotate the potential outlier.
food_maxfat0 = food_maxfat.iloc[0]
plt.annotate(food_maxfat0.Food, (food_maxfat0.Fat, food_maxfat0.Energy));

# I think it is a leverage point -- it may or may not be an outlier.


# In[14]:

# 3c. Use DFFITS to find outliers
infl = model1.get_influence()
dffits = infl.dffits[0]
dffits_threshold = 2 / np.sqrt(len(df_ff))
outliers = df_ff.ix[dffits > dffits_threshold, ['Food', 'Fat','Energy']]
outliers


# In[15]:

# 3d. Remove outliers and create a new regression model.
df_ff_clean = df_ff[~df_ff.index.isin(outliers.index)]
model2 = smf.ols('Energy ~ Fat', data=df_ff_clean).fit()
model2.summary()


# Comparison between the old and new models:
# - R-squared: changed from 0.700 to 0.724 (improved)
# - Coefficients:
#   - Fat: changed from 9.57 to 9.27
#   - Intercept: changed from 134.65 to 135.66
# - Confidence Interval of Coefficients:
#   - Fat: changed from [8.77, 10.37] to [8.54, 10.01]
#   - Intercept: changed from [122.96,146.34] to [124.92, 146.41]

# In[16]:

# 3e. Create a new scatterplot. Showing old an new regression lines.
xs = [0,55]
ys1 = model1.predict({'Fat': xs})
ys2 = model2.predict({'Fat': xs})
plt.scatter(df_ff_clean.Fat, df_ff_clean.Energy)
plt.scatter(outliers.Fat, outliers.Energy, s=50, color='r', marker='v')
plt.plot(xs, ys1, 'r--', label='Before outlier removal')
plt.plot(xs, ys2, 'k', label='After outlier removal')
plt.legend(loc='upper left')
plt.xlabel('Fat')
plt.ylabel('Energy')

