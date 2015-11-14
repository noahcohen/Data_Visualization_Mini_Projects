
# coding: utf-8

# # SI 370 - Lab Session 2: Basic Statistics
# 
# ## Objectives
# 1. Perform basic statistical analyses using Pandas objects
# 2. Build simple linear regression models
# 3. Identify and remove outliers
# 
# ## What to do
# 1. Download the `.ipynb` file of this notebook at `CTools->SI370->Resources->Week 3->Lab 2
# 2. Follow this notebook step-by-step. Execute every cell and verify the result as you go.
# 3. Do the exercises as instructed and fill in the answers on the lab worksheet.
# 4. For exercise questions, insert new cells into this notebook and execute them.
# 5. Take notes by adding Markdown cells or add comments to existing code. Keep this notebook for future reference.
# 6. Turn in the worksheet before you leave. Use the [online version of the worksheet](https://docs.google.com/document/d/1h7_CarxBKqrlUSkmpLz6lIBtRJifQRmUGRvWJ8sIeaI/edit) for after-class study.

# # 1. Setup Environment and Import Data

# ## 1.1 Install `statsmodels` package.
# `statsmodels` is a Python package that provides various statistical analysis tools. We will use `statsmodels` for this lab. The package does not come with `Anaconda` by default, but we can easily install it.
# 
# - Open `Terminal` (or `cmd` in Windows)
# - Choose an appropriate option:
#   - If python3 is your default python, then type
#         conda install statsmodels
#   - If python3 is NOT your default python, which means you had to activate your Python environment before each session (e.g., "`source activate python3`" on Mac or "`activate python3`" on Windows), then do the following
#         conda install --name=python3 statsmodels
#     where `python3` should be whatever the name of your conda environment is.
# - Follow the instructions to complete installation. Then come back to this notebook. The rest of the lab will be done within this notebook.

# ## 1.2 Import Data
# Let us start with necessary imports and data loading. Just execute every cell below. Remember: a convenient shortcut to run a cell and then jumps to the next cell is `Shirt + Enter`.

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:

# This is the Anscombe's quartet
# Source - Wikipedia: https://en.wikipedia.org/wiki/Anscombe%27s_quartet
from io import StringIO

TESTDATA=StringIO("""X1,Y1,X2,Y2,X3,Y3,X4,Y4
10,8.04,10,9.14,10,7.46,8,6.58
8,6.95,8,8.14,8,6.77,8,5.76
13,7.58,13,8.74,13,12.74,8,7.71
9,8.81,9,8.77,9,7.11,8,8.84
11,8.33,11,9.26,11,7.81,8,8.47
14,9.96,14,8.1,14,8.84,8,7.04
6,7.24,6,6.13,6,6.08,8,5.25
4,4.26,4,3.1,4,5.39,19,12.5
12,10.84,12,9.13,12,8.15,8,5.56
7,4.82,7,7.26,7,6.42,8,7.91
5,5.68,5,4.74,5,5.73,8,6.89""")

df = pd.DataFrame.from_csv(TESTDATA, index_col=None)
df


# This dataset contains 4 groups of data: `(X1, Y1)`, to `(X4, Y4)`.
# 
# It is not hard to notice that `X1`, `X2`, and `X3` are identical. But that doesn't matter now.
# 
# We can do a quick vis of the data by creating scatterplots.

# In[ ]:

fig, axs = plt.subplots(2, 2, figsize=(12,9))
for i, ax in enumerate(axs.flat):
    j = i + 1
    ax.scatter(df['X%d'%j], df['Y%d'%j])
    ax.set_title('(%d)'%j)
    ax.set_xlabel('X%d'%j)
    ax.set_ylabel('Y%d'%j)
    ax.grid(True)


# # 2. Compute Summary Statistics

# Pandas `DataFrames` and `Series` come with a couple of convenient functions for computing basic summary statistics. Try the following commands. Their meanings are quite self-explanatory.

# In[ ]:

df.mean()


# In[ ]:

df.median()


# In[ ]:

df.std()


# In[ ]:

df.max()


# In[ ]:

# This one computes the maximum along the 1st axies (i.e,. across columns).
df.max(axis=1)


# In[ ]:

df.var()


# You can also apply the functions on a column (i.e,. a Series), too. Such as...

# In[ ]:

df.X1.mean()


# In[ ]:

# This one is slightly more complicated... But you can figure this out easily.
(df.X1 + df.X2).mean()


# In[ ]:

# And finally...
df.describe()


# ### Exercise: Worksheet Problem 1###
# Complete the worksheet problem 1 using the above functions or any necessary combinations of them.

# # 3. Binning, Grouping, and Histograms
# To quickly understand the distribution of data, it is a good idea to use binning and grouping and creating histograms.
# 
# ## 3.1 Binning and Grouping
# 
# Use the following code to create bins based on `X1`'s values, and check the mean value of both `X1` and `Y1` within each bin.

# In[ ]:

# Since we have very few data points, I will only use 3 bins (by setting num = 4)
bins_by_x1 = np.linspace(start=3, stop=15, num=4)
groups_by_x1 = df[['X1','Y1']].groupby(pd.cut(df.X1, bins_by_x1))
groups_by_x1.mean()


# The above result shows the __mean__ value of `X1` and `Y1` binned by the value of `X1`.
# 
# To understand what is going on in the above commands, feel free to print out the intermediate variables, including:
# - `bins_by_x1`
# - `pd.cut(df.X1, bins_by_x1)`
# - `groups_by_x1`
# 
# Also try the following commands:

# In[ ]:

groups_by_x1.median()


# In[ ]:

groups_by_x1.size().to_frame(name='count')


# In addition to using existing aggregate functions (i.e., `max, min, mean, median, size`, etc.), you can also define custom functions to "`apply`" to the grouping object.
# 
# The following should generate the same result as the previous one. Try to figure out how it works.

# In[ ]:

groups_by_x1.apply(lambda x: len(x)).to_frame(name='count')


# In[ ]:

# Or, alternatively, and more confusingly ...
groups_by_x1.apply(lambda x: pd.Series({'count': len(x)}))


# ### Exercise: Worksheet Problem 2
# - For `X4`, create 4 bins (`num=5`), starting at 6 (`start=6`), ending at 20 (`stop=20`).
# - Group the values of `X4` and `Y4` into the bins.
# - Use appropriate aggregate functions to fill in the blanks in Worksheet Problem 2.
# - Verify your results with the scatterplot above.

# ## 3.2 Creating Histograms
# To create a histogram, simply use `.hist()` on a `Series` object. It automatically handles binning, grouping, and counting.
# 
# By default, `hist()` generates 10 bins. You may customize this by specifying the `bins` parameter, for example, `hist(bins=5)`.
# 
# Try the following.

# In[ ]:

df.Y1.hist(bins=5)


# you may call `hist()` on a `DataFrame` object and obtain a panel of histograms, one for each individual column.

# In[ ]:

_ = df[['Y1','Y2','Y3','Y4']].hist(bins=5, figsize=(10,6))


# In the above example, notice two things:
# - I used `figsize=(10, 6)` to specify the size of the figure (the unit is inch, although the `%matplotlib inline` configuration reduces the figure sizes by a predefined ratio automatically).
# - I put "`_ =`" in the front to avoid seeing the returned value of `df.hist`, which I do not care. If you are curious, it is an array of `axes` objects of `matplotlib`. 
#     - In general, `_` can be used whenever you want to ignore the return value of a function.

# You can also make histograms that have side-by-side bars for multiple variables. For the following example, I used `matplotlib`'s `hist` function instead of the equivalence of `pandas`, because it is easier to use the former one to make side-by-side histograms like this.

# In[ ]:

bins = np.linspace(start=2, stop=12, num=6)
plt.hist(df[['Y1','Y2','Y3','Y4']].values, 
         bins, 
         label=['Y1','Y2','Y3','Y4'])
plt.legend(loc="upper left")
plt.grid(True)


# ### Exercise: Worksheet Problem 3
# - Create histograms for `X1` and `X4` respectively. 
# - Answer Worksheet Problem 3 based on the histograms.

# ## 3.3 Boxplots
# Boxplots (or box and whisker diagrams) is a another good way of depicting the distribution of numerical data by showing the mean, min, max, and interquantile range(IQR).
# 
# Using `pandas`, it is very easy to create a boxplot of multiple variables.

# In[ ]:

_ = df[['X1','Y1']].boxplot()


# ### Exercise: Worksheet Problem 4
# - Make the boxplot for variables `X3, Y3, X4, Y4`.
# - Can you detect anomaly by just reading the boxplot?
#     - Report in the worksheet: for which variable(s) did you find outliers?

# # 4. Correlation and Covariance
# The covariance and correlation between two variables can be computed using `cov` and `corr` functions.

# In[ ]:

# Covariance between X1 and Y1
df.X1.cov(df.Y1)


# In[ ]:

# Pearson's correlation between X1 and Y1
df.X1.corr(df.Y1)


# In[ ]:

# We can also calculate the correlation matrix
np.corrcoef(df.X1, df.Y1)


# In[ ]:

# This generates the same result, as a DataFrame.
df[['X1','Y1']].corr()


# In[ ]:

# This is the covariance matrix.
df[['X1', 'Y1']].cov()


# In[ ]:

# Visualize the correlation matrix of all variables.
corr_matrix = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, cmap=plt.cm.OrRd)
fig.colorbar(cax)

tags = ['']+corr_matrix.columns.tolist()
ax.set_xticklabels(tags)
ax.set_yticklabels(tags)
plt.title('Correlation Matrix', y=-0.1)


# ### Exercise: Worksheet Problem 5
# - Solve problem 5 using appropriate functions.

# ## 5. Basic Hypothesis Testing
# In this section, let's treat `X1`, `Y1`, ..., `X4`, `Y4` as independent samples from different distributions.

# ## 5.1 _t_-test
# If you want to know whether there is a significant difference between the means of two samples, you should use a _t_-test.
# 
# There are multiple types of _t_-tests. We focus on __Welch's _t_-test__, a  kind of independent two-sample _t_-test. It is a two-sided test for the null hypothesis that 2 independent samples have identical average values, which does not assume equal variance of the two samples.
# 
# The following quote from [scipy documentation](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_ind.html) is helpful for understand what this test does.
# 
# > We can use this test, if we observe two independent samples from the same or different population, e.g. exam scores of boys and girls or of two ethnic groups. The test measures whether the average (expected) value differs significantly across samples. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores. If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
# 
# Performing the actual test is simple. Try the following command, it tests the null hypothesis that X1 and X4 have identical average values, without assuming equal variance between the two samples (`equal_var = False`).

# In[ ]:

from scipy.stats import ttest_ind
ttest_ind(df.X1, df.X4, equal_var=False)


# It should return a tuple, with the following two elements:
# - _t_: the calculated t-statistic
# - _p_: the two-tailed _p_-value.
# 
# Usually we use a threshold of 0.05 or 0.1 for the _p_-value. If the _p_-value is greater than the threshold we set, then we "fail to reject the null hypothesis", which indicates that there is no _statistically_ significant difference between the means of the two samples.
# 
# Look at the above test result, what is the _p_-value? Compare it against 0.05. Is there a significant difference between the means of `X1` and `X4` significant?

# In[ ]:

# Let's try a pair of made-up "samples."
# What is the p-value this time? Is it significant?
ttest_ind([1,2,3],[-1,-2,-3], equal_var=False)


# ### Exercise: Worksheet Problem 6
# - Step 1: Generates two additional random samples, X5 and X6 (code provided).
# - Step 2: Plot the density distributions of X5 and X6 (code provided).
# - Step 3: Perform a t-test for the null hypothesis that X5 and X6 have identical mean value, assuming non-equal variance. _p_-value threshold set to 0.05. Fill in the blanks in Worksheet Problem 6.
# - __(Optional)__ Compare the result to a similar t-test assuming equal variance (`equal_var=True`), and see if there is any difference.

# In[ ]:

# Step 1. Generate two additional samples from normal distributions.
np.random.seed(0)
X5 = np.random.randn(100) * 2 + 0.3
X6 = np.random.randn(100) - 0.15


# In[ ]:

# Step 2. Plot the distribution of these two variables
pd.DataFrame({'X5':X5,'X6':X6}).plot(kind='kde', grid=True)


# ## 5.2 Normality test
# A number of commonly used statistical models or diagnostic statistics assume normal distributions (and/or equal variance of the samples), but many real-world data are normally distributed. [Normality tests](https://en.wikipedia.org/wiki/Normality_test) are used to determine if a dataset or a sample is normally distributed.
# 
# There are many ways of doing normality tests, we focus on two ways:
# - [normal probability plot](https://en.wikipedia.org/wiki/Normal_probability_plot), a special case of [QQ-plots (quantile-quantile plot)](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot), an intuitive way of checking the similarity of two distributions.
# - [D'Agostino and Pearson's omnibus test](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.normaltest.html), a test for the null hypothesis that a sample comes from a normal distribution.

# In[ ]:

# Plot normal probability plot (QQ-plot) for X5, which is generated above.
from scipy.stats import probplot
probplot(X5, plot=plt)
plt.grid(True)


# In the above plot, if all the regression line fits the data points well, it indicates that this sample is likely drawn from a normal distribution.
# 
# Below is an example where the regression line does not fit well, indicating the data is not normally distributed.

# In[ ]:

probplot(df.X4, plot=plt)
plt.grid(True)


# In[ ]:

# D'Agostino and Pearson's test for X5.
from scipy.stats import normaltest
normaltest(X5)


# Similar to the previous test we introduced, the `normaltest()` function returns a tuple of:
# - k2: a statistic that combines skew and kurtosis of the distribution
# - _p_-value: a 2-sided chi squared probability for the hypthesis test, what we care about.

# ### Exercise: Worksheet Problem 7
# - Make normal probability plots for X1, Y1, Y3, and Y4. 
#   - Are they normally distributed?
# - Perform normality test on the above variables. 
#   - Are the _p_-values significant?
#   - Report your findings in WorkSheet Problem 7.

# # 6. Linear Regression
# Let's build a linear regression model of Y1 from X1, which means:
# - Y1 is the __dependent variable__ (a.k.a., regressand, endogenous variable, response variable, criterion variable).
# - X1 is the __independent variable__ (a.k.a., regressor, exogenous variable, explanatory variable, covariate, input variable, or predictor). See [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression) for more.

# In[ ]:

# In order to correctly import these,
# you should install statsmodels (see instructions on the top of this notebook)
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:

# ignore the warning.
model1 = smf.ols('Y1 ~ X1', data=df).fit()
model1.summary()


# In[ ]:

# Plot the data with a regression line.
# The regression line is generated by model1, which we just trained.
plt.scatter(df.X1, df.Y1)
xs = np.linspace(3, 15, 2)
ys_predicted = model1.predict({'X1': xs})
plt.plot(xs, ys_predicted)
plt.grid(True)


# ### Exercise: Worksheet Problem 8
# - Train a linear regression model with `Y3` as the __response__ and `X3` as the __predictor__. Save it as `model2`.
#   - Use `model2.summary()` to answer the questions in the worksheet's Problem 8.
#   - Plot the data of (X3, Y3) with a regression line using `model2`.
#   - __(Optional)__: identify and remove the outlier from (X3, Y3), and retrain a regression model. Compare the difference of the regression lines before and after filtering the outlier.

# # Turn in the worksheet before you leave.
# For self-studying after the lab, use the [online version of the worksheet](https://docs.google.com/document/d/1h7_CarxBKqrlUSkmpLz6lIBtRJifQRmUGRvWJ8sIeaI/edit).

# # References
# - [Linear regression with Python](http://connor-johnson.com/2014/02/18/linear-regression-with-python/)
# - [Linear regression in python-statsmodels](http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html)
# - [What value of an R-squared score is good](http://people.duke.edu/~rnau/rsquared.htm)
