
# coding: utf-8

# # SI 370 - Lab Session 3: Vis Basics
# ## Objectives:
# 1. Practice creating various plots for nominal, ordinal, and quantitative data using `matplotlib`, `pandas`, and `seaborn`.
# 3. Understand the limitation of built-in plotting functionality of `pandas` and when to switch to `matplotlib` and when to use `seaborn`.
# 4. Understand the existance of outliers and how to detect and remove outliers (left-over topic from the last week).
# 
# ## What to do
# 1. Download the `.ipynb` file of this notebook.
# 2. Follow this notebook step-by-step. Execute every cell and verify the result as you go.
# 3. Do the exercises as instructed and fill in the answers on the lab worksheet.
# <br><span style="color:red">When you are doing the exercises, do not edit existing cells. __Insert new cells__ below the exercise questions, and type or copy over the code into the new cells.</span>
# 5. Take notes by adding Markdown cells or add comments to existing code. Keep this notebook for future reference.
# 6. Turn in the worksheet before you leave. Use the [online version of the worksheet](https://drive.google.com/open?id=1DtbBEKTwxE2Ji8IrCQZjP6jmgRazhhMcHCHa6t0TM6k) for after-class study.

# ## 1. Install Seaborn
# `Seaborn` is a python package built on top of `matplotlib`, that provides additional (awesome) style management and a variety of convenient plotting functions. It can also work closely with `DataFrames` of `pandas`.
# 
# Before everything, install `seaborn` in your `Anaconda` environment.
# 
# In `Terminal` (or `Command Prompt` in Windows), type in the following command, just as if you would install any other package.
#         
#         conda install seaborn
# or
#         
#         conda install --name=python3 seaborn
# 
# When you are done, come back to this IPython notebook. The rest of the lab will be done here.

# In[1]:

# First things first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[3]:

# Let's first plot something using matplotlib's default style
data = np.random.randn(2, 100)  # Generate 100 random points in 2-D space
plt.scatter(data[0], data[1], color='b');


# <span style="color:red">If you get an import error in the following cell, you can restart the IPython kernel by going to the menu at the top and select: Kernel -> Restart and click on the "Restart" button.</span>

# In[4]:

# Now let's import seaborn
# See how it beautifies the style automatically?
# If you get an error executing this cell, see the red note above.
import seaborn as sns
sns.set(color_codes=True)
plt.scatter(data[0], data[1], color='b');


# Notice how the plot with the same data looks much nicer now?
# 
# In the second line of the above cell, the option `color_codes=True` allows seaborn to automatically change the interpretation of shorthand codes like "r" (red) or "b" (blue) by matplotlib in subsequent plots. This will result in more "pretty" colors aesthetically.

# ## 2. Download and Import Data
# For this lab, we will use the movies dataset. 
# 
# __Download the movies dataset from CTools:__
# 
# - Use [this link](https://ctools.umich.edu/access/content/group/09aeadb1-5ac1-4433-9741-033966380c8a/Datasets/movies.csv) or go to `CTools -> SI 370 -> Resource -> Datasets -> movies.csv` to download the dataset. Once downloaded, put it in the same folder with this notebook.

# In[5]:

# Import the dataset
df = pd.read_csv('movies.csv')
df.head()


# In[6]:

# Take a look at its shape
df.shape


# In[77]:

# Some cleaning - You don't need to know the details here.

# 1. string to number (some "Unknowns" in these columns
#    will be replaced with NaN)
cols = ['US Gross', 'Worldwide Gross']
df[cols] = df[cols].convert_objects(convert_numeric=True)

# 2. string to datetime (some cells with different formats 
#    will be replaced with NAT)
rdate = pd.to_datetime(df['Release Date'], format='%d-%b-%y', coerce=True)
rdate[rdate > '2015-01-01'] -=  np.timedelta64(100, 'Y')
df['Release Date'] = rdate


# In[78]:

# Take another look at its shape.
df.shape


# ## 3. Visualizing 2D Quantitative Data

# ### 3.1 Scatter Plot

# In[9]:

# Visualize Rotten Tomato vs. IMDB Rating
# There are three solutions: Pandas, Matplotlib, and Seaborn.
# Solution 1: Pandas
df.plot(x='Rotten Tomatoes Rating', y='IMDB Rating', kind='scatter');


# In[11]:

# Solution 2: matplotlib
# In Matplotlib, we have to manually specify the xlabel and ylabel.
# However, plotting this way gives us more flexibility and control.
plt.scatter(x=df['Rotten Tomatoes Rating'], y=df['IMDB Rating'])
plt.xlabel('Rotten Tomatoes Rating')
plt.ylabel('IMDB Rating');


# In[12]:

# Solution 3: using Seaborn
# Note: regplot means "Regression Plot". 
# Setting fit_reg=False means we don't want a regression line.
sns.regplot('Rotten Tomatoes Rating', 'IMDB Rating', data=df, 
            fit_reg=False);


# With seaborn, it is also convenient to leverage the color (_hue_) channel to have additional encoding of __nominal data__ in the scatter plot. See the following example.

# In[13]:

# Adding "hue" to the scatter plot.
# This is a slightly different function:
#   lmplot means "linear model plot".
# Note that we have to filter the distributors, 
# otherwise we would end up with too many colors.
few_distributors = ['Universal', 'Walt Disney Pictures', 'MGM']
df_few_distributors = df[df['Distributor'].isin(few_distributors)]
sns.lmplot('Rotten Tomatoes Rating', 'IMDB Rating', 
           data=df_few_distributors,
           hue='Distributor', fit_reg=False);


# In[14]:

# Another example scatter plot, depicting gross vs budget per film.
few_genres = ['Drama','Adventure','Comedy']
df_few_genres = df[df['Major Genre'].isin(few_genres)]

# We grab the returned value of lmplot so that we can alter the plotted
# figure later on.
grid = sns.lmplot('Production Budget', 'Worldwide Gross',
                  data=df_few_genres, hue='Major Genre', fit_reg=False)

# We alter the axes of the figure to use a log-log scale.
# This helps us see the distributor of the points better.
# Try plotting the same figure with and without the following line.
grid.set(xscale="log", yscale="log", xlim=(1e4,1e9), ylim=(1e5, 2e9));


# ### Exercise:
# 1. Create a scatter plot of `IMDB Rating` vs. `Wordwide Gross`.
# 2. Add `hue="MPAA Rating"` to the scatter plot. Limit to `R, PG-13, and G` only.
# 3. Answer the questions in __Worksheet Problem 1__ by reading the scatter plot.

# In[15]:

sns.lmplot('Worldwide Gross','IMDB Rating', 
           data=df[df['MPAA Rating'].isin(('R','PG-13','G'))],
           hue='MPAA Rating', fit_reg=False)


# ### 3.2 Line Chart
# Commonly used to visualize a trend in data over time; thus one of the variables should be time or a time-like measurement.

# In[16]:

# Let's simulate a "random walk".
# Run this cell multiple times. Each time, you will get a different result.

# Each step: we flip a coin, making a decision to either move forward (+1), backward (-1), or stay (0)
decisions = np.random.randint(-1, 2, 1000)

# Using np.cumsum() we can conveniently calculate "where we are now".
locations = np.cumsum(decisions)

# Visualize the locations over time (or steps)
# The semicolon is to suppress return of the plot function
plt.plot(locations);


# In[17]:

# We can repeat the above random walk multiple times, and overlay the results together
for i in range(10):
    decisions = np.random.randint(-1, 2, 1000)
    locations = np.cumsum(decisions)
    plt.plot(locations, label='Walk %d'%i)

# This line alters the legend location.
# Otherwise the legend would endup overlap with the lines.
plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5));


# In[18]:

# Visualize the trend of the average movie budget over years.

# Step 1: prepare the data.
# Note: the first line extracts years from release dates.
years = pd.DatetimeIndex(df['Release Date']).year
by_year = df.groupby(years)
budget_mean_by_year = by_year[['Production Budget']].mean()

# Step 2: Make the plot
budget_mean_by_year.plot()

# Step 3: Limit the x-axis range
plt.xlim(1950, 2010)


# ### Exercise:
# 1. Create a line chart depicting the trend of worldwide gross over years.
# 2. Limit the range on x-axis to be `(1950, 2010)`.
# 3. Answer the question in __Worksheet Problem 2__ using the chart.

# In[23]:

# Step 1: Prepare the data
years = pd.DatetimeIndex(df['Release Date']).year
by_year = df.groupby(years)
budget_mean_by_year = by_year[['Worldwide Gross']].mean()

# Step 2: Make the plot
budget_mean_by_year.plot()
plt.xlim(1950,2010)
plt.ylim(0, 2e+8)


# ## 4. Visualizing 2D  Data (1 Nominal + 1 Quan.)

# ### 4.1 Bar Plot

# In[28]:

# Using pandas to create a bar plot
by_distributor = df.groupby('Distributor')
mean_gross = by_distributor[['Worldwide Gross']].mean()
top_mean_gross = mean_gross.sort('Worldwide Gross', ascending=False).head(10)
top_mean_gross.plot(kind='barh');


# In[29]:

# Use seaborn to create a bar plot of mean value with confidence intervals
sns.barplot(x='MPAA Rating', y='Worldwide Gross', data=df);


# In[30]:

# Adding a hue channel to the bar plot
# Compare two distributors only
df_universal_disney = df[df.Distributor.isin(('Universal', 'Walt Disney Pictures'))]
sns.barplot(x='MPAA Rating', y='Worldwide Gross', 
            data=df_universal_disney,
            hue='Distributor');


# ### 4.2 Box Plot

# In[31]:

# Create a box plot, using the same data as above
sns.boxplot(x='MPAA Rating', y='Worldwide Gross', 
            data=df_universal_disney,
            hue='Distributor');


# ### Exercise:
# 1. Create a __bar plot__ comparing the __mean production budget__ of films by `20th Century Fox, Universal, Walt Disney Pictures`.
# 2. Add `hue='Major Genre'` to the bar plot. Limiting to `Drama`, `Adventure`, and `Thriller/Suspense` only.
# 3. Create a __box plot__ of the same data.
# 4. Answer the questions in __Worksheet Problem 3__.

# In[2]:

df_few_distributors = df[df.Distributor.isin((
            '20th Century Fox','Universal','Walt Disney Pictures'))]
df_few_genres = df_few_distributors[df_few_distributors['Major Genre'].isin(
            ('Drama', 'Musical', 'Thriller/Suspense'))]
sns.barplot(x='Distributor', y='Production Budget', 
            data=df_few_genres,
            hue='Major Genre');


# In[33]:

sns.boxplot(x='Distributor', y='Production Budget', 
            data=df_few_genres,
            hue='Major Genre');

plt.ylim(0,1.3e+8);


# ## 5. Stacked Bar Chart

# In[35]:

# A simple example of stacked bar charts.
df_toy = pd.DataFrame({'A':[1,2,3], 'B':[2,3,1]}, index=list('xyz'))
df_toy.plot(kind='bar', stacked=True);


# In[1]:

# A more complex example or stacked bar charts. 
# Comparing the average contribution of each genre to the world-wide gross by distributor 
# over the years of 1991-2010.

# Step 1: Prepare the data
years = pd.DatetimeIndex(df['Release Date']).year
df_in_years = df[(years >= 1991) & (years <= 2010)]
top_dstr = ['Summit Entertainment', '20th Century Fox', 
                    'Warner Bros.', 'Walt Disney Pictures', 
                    'Dreamworks SKG', 'Paramount Pictures',  
                    'Universal', 'Sony Pictures']
df_top_dstr = df_in_years[df_in_years.Distributor.isin(top_dstr)]
df_genre_pivot = df_top_dstr.pivot_table(
                    values='Worldwide Gross', columns='Major Genre', 
                    index='Distributor', aggfunc=np.mean)

# Step 2: Make the plot
df_genre_pivot.plot(kind='barh', stacked=True);


# In[37]:

# Use a custom palette to avoid repeating the colors.
with sns.hls_palette(12):
    df_genre_pivot.plot(kind='barh', stacked=True);

# Add a label to the x-axis
plt.xlabel('Mean Worldwide Gross');


# ### Exercise:
# 1. Create a stacked bar chart, comparing the __Total Production Budget__ of the films by top distributors over the years of 1991-2010. Using __MPAA Rating__ as hue. <br>_Hint_: You may reuse `df_top_dstr` create above. Start with creating a pivot_table.
# 2. Answer the questions in __Worksheet Problem 4__.

# In[38]:

df_rating_pivot = df_top_dstr.pivot_table(
    values='Production Budget', index='Distributor', 
    columns='MPAA Rating', aggfunc=np.sum)

df_rating_pivot.plot(kind='barh', stacked=True);

plt.xlabel('Total Production Budget');


# ## 6. Histogram and Kernel Density Plot

# In[39]:

# Create a histogram using Pandas
df['IMDB Rating'].plot(kind='hist', bins=20, title='IMDB Rating');


# In[40]:

# Create a histogram using seaborn with kernel density estimation (KDE)
sns.distplot(df['IMDB Rating'].dropna(), kde=True, bins=20);


# ### Exercise:
# 1. Create a histogram of Rotten Tomatoes Rating for all films.
# 2. Add KDE to the above histogram.
# 3. Answer the question in __Worksheet Problem 5__.

# In[41]:

sns.distplot(df['Rotten Tomatoes Rating'].dropna(), kde=True, bins=20);

plt.xlim(0,100);


# ## 7. Scatter Plot Matrix (SPLOM)

# In[3]:

# SPLOM example
# This function needs a clean DataFrame.
df_cleaned = df[['Worldwide Gross','Production Budget','IMDB Rating',
                 'MPAA Rating']].dropna()
sns.pairplot(df_cleaned);


# In[43]:

# with hue
sns.pairplot(df_cleaned, hue='MPAA Rating');


# ### Exercise:
# 1. __[Code Provided]__ Create a new DataFrame with only Warner Bros. and Sony Pictures.
# 2. Use to the DataFrame to create a SPLOM, visualizing the relations between `Wordwide Gross, Production Budget, IMDB Rating`, and `Rotten Tomatoes Rating`, using __Distributor__ as hue.
# 3. Answer the question in __Worksheet Problem 6.__

# In[46]:

# Step 1. (Code Provided)
df_two_dstr = df[df.Distributor.isin(('Warner Bros.', 'Sony Pictures'))]


# In[47]:

# Step 2. Clean and plot
# Takes can take quite a while.
df_cleaned2 = df_two_dstr[['Worldwide Gross', 'Production Budget',
                           'IMDB Rating', 'Rotten Tomatoes Rating',
                           'Distributor']].dropna()

sns.pairplot(df_cleaned2, hue='Distributor');


# ## 8. Scatter Plot with Fitted Curve

# In[48]:

# Create a scatter plot with a linear fitted line.
sns.regplot(x='Rotten Tomatoes Rating', y='IMDB Rating', data=df);


# In[49]:

# Another example with a linear fit -- with the Anscombe's Quartet data
df_ans = sns.load_dataset("anscombe")
df_ans2 = df_ans.loc[df_ans.dataset == "II"]
sns.regplot(x="x", y="y", data=df_ans2);


# In[50]:

# Fit a higher-order polynomial regression
sns.regplot(x="x", y="y", data=df_ans2, order=2);


# In[51]:

# Let's plot them side-by-side
fig, axes = plt.subplots(ncols=2, figsize=(12,5), sharey=True)
sns.regplot(x="x", y="y", data=df_ans2, 
            ax=axes[0], scatter_kws={"s": 100})
sns.regplot(x="x", y="y", data=df_ans2, order=2, 
            ax=axes[1], scatter_kws={"s": 100});


# ### Exercise:
# 1. __[Code Provided]__ Get the Anscombe's Dataset III. 
# 2. Create two regression plots side-by-side, with linear and 2nd order fitted lines respectively.
# 3. There is no worksheet question for this problem.

# In[52]:

# Step 1. [Code Provided] Get Ansombe's dataset III
df_ans3 = df_ans[df_ans.dataset=='III']


# In[54]:

# Step 2. Plot
fig, axes = plt.subplots(ncols=2, figsize=(12,5), sharey=True)
sns.regplot(x="x", y="y", data=df_ans3, ax=axes[0], scatter_kws={"s": 100})
sns.regplot(x="x", y="y", data=df_ans3, order=2, ax=axes[1], scatter_kws={"s": 100});
axes[0].set_ylim(2, 15);


# ## 9. Scatter Histograms

# In[55]:

# Create a scatter plot with marginal histograms.
sns.jointplot(x='Rotten Tomatoes Rating', y='IMDB Rating', data=df);


# In[56]:

# Add regresion lines
sns.jointplot(x='Rotten Tomatoes Rating', y='IMDB Rating', 
              data=df, kind='reg');


# ### Exercise:
# 1. Create a scatterplot with marginal histograms with regression lines for Worldwide Gross vs. Production Budget.
# 2. There is no worksheet problem for this question.

# In[57]:

sns.jointplot(x='Production Budget', y='Worldwide Gross',
              data=df, kind='reg');


# ## Summary
# What we have learned so far:
# - Seaborn is a package that provides professional-looking style and various convenient plotting functions.
# - Adding `hue` to seaborn plots is a convenient way of encoding an extra nominal data dimension in the visualization.
#   - In practice, we may need to do grouping and filtering to reduce the amount of information encoded in the `hue` channel.
# - In order to produce complex visualizations, we sometimes need to go through multiple steps of data filtering and aggregation. Often, the [split-apply-combine routine](http://pandas.pydata.org/pandas-docs/stable/groupby.html) is necessary. To perform this routine, we may use `pivot_table()`.

# ## 10. Outlier Detection and Removal
# This is a left-over topic from the last week.

# In[58]:

# Visualize the Anscombe's Quartet again.
fig, axs = plt.subplots(2, 2, figsize=(8, 7))
rome_nums = ['I','II','III','IV']
for i, ax in enumerate(axs.flat):
    j = i + 1
    df_i = df_ans[df_ans.dataset==rome_nums[i]]
    df_i.plot('x', 'y', kind='scatter', ax=ax)
    ax.set_xlabel('X%d'%j)
    ax.set_ylabel('Y%d'%j)


# In[59]:

# Create a linear regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf
df_ans3 = df_ans[['x','y']][df_ans.dataset == 'III'].reset_index(drop=True)
df_ans4 = df_ans[['x','y']][df_ans.dataset == 'IV'].reset_index(drop=True)
model1 = smf.ols('y ~ x', data=df_ans3).fit()
model1.summary()


# In[60]:

# Find outliers using Cook's Distance
influence = model1.get_influence()
cooks_distance = influence.cooks_distance[0]
number_of_observations = len(df_ans3)

# Use an empirical threshold value
cooks_threshold = 4 / number_of_observations

# Show outliers
df_ans3[cooks_distance > cooks_threshold]


# In[61]:

# Visualize Cook's Distance
# It is very obvious which data point is an outlier.
plt.stem(cooks_distance);
plt.xticks(range(len(df_ans3)));
plt.ylim(0, 1.5);


# In[62]:

# Find outliers using DFFITS
number_of_observations = len(df_ans3)
number_of_parameters = 2  # parameters include: intercept, x
dffits = influence.dffits[0]

# Use an empirical threshold
dffits_threshold = 2 * np.sqrt(number_of_parameters / number_of_observations)
df_ans3[np.abs(dffits) > dffits_threshold]


# In[63]:

# Visualize DFFITS (absolute value)
plt.stem(np.abs(dffits));
plt.xticks(range(len(df_ans3)));


# ### Some Background Knowledge on Outlier Detection
# We will use the following influence statistics for detecting outliers.
# - __Cook's distance (`cooks_d`)__: measures the distance between the fitted values (Y) calculated with and without a given observation point.
# - __DFFTIS (`dffits`)__: same as Cook's distance, except the distance is measured by the approximate number of standard deviations.
# 
# * * * 
# 
# In practice, the following __empirical cut-off thresholds__ may be used to identify outliers or leverage points. Let us define n as the number of observations, and p as the number of parameters.
# - __Cook's distance__: 4/n, or simply, 1. Observation points whose Cook's distance is greater than this threshold are considered as influential points.
# - __DFFITS__: 2 * sqrt(p/n). Observation points whose absolute DFFITS values are greater than  this threshold are considered influential points.

# In[64]:

# Remove outliers
# using Cook's Distance
outlier_criterion = (cooks_distance > cooks_threshold)
outlier_indexes = np.nonzero(outlier_criterion)[0]

df_ans3_cleaned = df_ans3[~outlier_criterion]
print("%d points before removal; %d after removal."%(
        len(df_ans3), len(df_ans3_cleaned)))


# In[65]:

# Train a new model with outliers removed
model2 = smf.ols('y ~ x', data=df_ans3_cleaned).fit()
model2.summary()


# In[66]:

# We are interested in how much the R-squared has improved.
print('Before outlier removal, R^2 = %f'%(model1.rsquared_adj))
print('After outlier removal, R^2 = %f'%(model2.rsquared_adj))


# In[67]:

# Plot the old and new regression lines together
xs = np.linspace(3, 15, 2)
ys1 = model1.predict({'x': xs})
ys2 = model2.predict({'x': xs})
plt.scatter(df_ans3_cleaned.x, df_ans3_cleaned.y, 
            s=100, marker='v')
plt.scatter(df_ans3.ix[outlier_indexes, 'x'], 
            df_ans3.ix[outlier_indexes, 'y'], 
            s=100, color='r', marker='v')
plt.plot(xs, ys1, 'r--', label='Before outlier removal')
plt.plot(xs, ys2, 'k', label='After outlier removal')
plt.legend(loc='upper left')
plt.xlabel('X3')
plt.ylabel('Y3')
plt.grid(True)


# ### Exercise:
# - Step 1: __[Code Provided]__ Generate a toy dataset with X, Y
# - Step 2: __[Code Provided]__ Create a scatter plot of (X, Y)
# - Step 3: Build a linear regression model for Y ~ X
# - Step 4: Identify outliers using Cook's distance and DFFITS
# - Step 5: Remove outliers and plot a new regression line
# - Step 6: Fill in the blanks in the __Worksheet Problem 7.__

# In[68]:

# Step 1. [CODE PROVIDED] Generate a toy dataset
np.random.seed(0)
X = np.arange(23)
Y = X * 2 + 3 + np.random.randn(len(X))

# Add artificial outliers
X = np.insert(X, 5, 10)
Y = np.insert(Y, 5, 55)
X = np.insert(X, 10, 3)
Y = np.insert(Y, 10, 30)


# In[69]:

# Step 2. [CODE PROVIDED] Scatter plot of X and Y
plt.scatter(X, Y, s=50)
plt.xlabel('X')
plt.ylabel('Y')


# In[73]:

# Steps 3-4
# Build a linear regression model and find outliers using DFFITS

# Build the linear regression model
df_xy = pd.DataFrame({'x':X, 'y':Y})
model1 = smf.ols('y ~ x', data=df_xy).fit()
influence = model3.get_influence()

# Calculate DFFITS
number_of_observations = len(df_xy)
number_of_parameters = 2  # parameters include: intercept, x
dffits = influence.dffits[0]

# Use an empirical cut-off threshold to find the outliers
dffits_threshold = 2 * np.sqrt(number_of_parameters / number_of_observations)
df_xy[np.abs(dffits) > dffits_threshold]


# In[76]:

# Step 5
# Remove outliers and plot a new regression line
outlier_criterion = (np.abs(dffits) > dffits_threshold)
outlier_indexes = np.nonzero(outlier_criterion)[0]

df_xy_cleaned = df_xy[~outlier_criterion]
model2 = smf.ols('y ~ x', data=df_xy_cleaned).fit()

xs = np.linspace(-3, 23, 2)
ys1 = model1.predict({'x': xs})
ys2 = model2.predict({'x': xs})
plt.scatter(df_xy_cleaned.x, df_xy_cleaned.y, 
            s=100, marker='v');
plt.scatter(df_xy.ix[outlier_indexes, 'x'], 
            df_xy.ix[outlier_indexes, 'y'], 
            s=100, color='r', marker='v');
plt.plot(xs, ys1, 'r--', label='Before outlier removal');
plt.plot(xs, ys2, 'k', label='After outlier removal');
plt.legend(loc='upper left');
plt.xlabel('X');
plt.ylabel('Y');

