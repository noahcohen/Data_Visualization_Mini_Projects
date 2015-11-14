
# coding: utf-8

# # Homework 7 Partial Solution
# 
# Notes:
# - __This is NOT a sample homework submission.__ The purpose of this notebook is for you to verify the correctness of your results (programming portion only). We post selected excellent homework submissions separately at `CTools -> Resources -> Excellent Homework Submissions`. 
# - The solutions provided here are for reference only. It is very likely that more than one solutions exist for a problem.
# - If you think there is any error in this notebook, please email `si370-staff@umich.edu`.

# In[1]:

import warnings
warnings.filterwarnings('ignore')


# In[2]:

import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
import scipy.cluster.hierarchy as sph
import sklearn as sk
import sklearn.metrics as skm
import sklearn.cluster as skc
import sklearn.decomposition as skd
import sklearn.mixture as skmix
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set(style='white', font_scale=1.3, color_codes=True)


# In[3]:

# 1a. Load data
from io import StringIO

df_raw = pd.DataFrame.from_csv(StringIO(
"""CustomerID,Gender,Payment,FavoriteProduct,\
Age,Income,HouseholdSize,Sales,StoreVisit
121,Female,Cash,Beer,42,65000,4,20,1
123,Male,Cash,Spirit,60,45000,2,300,10
154,Male,Debit,Beer,21,21000,1,46,18
166,Male,Cash,Spirit,72,48000,2,290,11
170,Female,Credit,Wine,36,90000,5,190,4
198,Female,Debit,Wine,34,85000,4,180,6
199,Male,Credit,Spirit,41,75000,2,15,1
205,Male,Cash,Wine,26,65000,1,20,2
222,Male,Cash,Spirit,21,21000,1,75,21
239,Female,Credit,Spirit,35,75000,4,200,5
268,Male,Debit,Spirit,68,44000,2,280,13
293,Male,Debit,Beer,23,30000,1,60,22
332,Male,Cash,Wine,65,55000,2,320,12
334,Male,Cash,Beer,22,15000,1,55,21
335,Female,Cash,Beer,24,23000,2,55,19
384,Female,Credit,Wine,27,125000,5,220,5
410,Female,Cash,Spirit,45,88000,3,15,1
420,Male,Cash,Beer,26,15000,2,20,1
537,Female,Debit,Beer,23,22000,2,50,16
584,Female,Cash,Beer,25,20000,2,60,20"""), index_col='CustomerID')

df_raw.head()


# In[4]:

# 1b. Filter data
df_filtered = df_raw[df_raw.StoreVisit >= 3]
df_filtered.shape


# In[5]:

# 1c. Create dummy variables
nominals = ['Gender', 'Payment', 'FavoriteProduct']
df_dummy = df_filtered.copy()
for n in nominals:
    for v in df_filtered[n].unique():
        df_dummy[v] = np.where(df_dummy[n] == v, 1, 0)
df_dummy.drop(nominals, axis=1, inplace=True)
df_dummy.shape


# In[6]:

# 1d. Normalize
df_norm = df_dummy.apply(lambda x: (x - x.mean()) / x.std())
df_norm.head()


# In[7]:

# 1e. Hierarchical Clustering
dist = spd.squareform(spd.pdist(df_norm, metric='euclidean'))
Z = sph.linkage(dist, method='single')
_ = sph.dendrogram(Z, labels=df_norm.index)
plt.gca().yaxis.grid(True)
plt.gca().set_yticks(range(10));


# A good distance threshold can be 6.2. We can observe many mergers happening at around 5 - 6, and after 6, there is a gap until the next merge. This threshold gives 3 clusters. If we selected a threshold from another "gap", i.e., 3 to 5, we would end up with too many small clusters, which will probably break the overall clustering structure of the dataset.

# In[8]:

# (1e continued)
# List cluster members
cluster_labels = sph.fcluster(Z, 6.2, criterion='distance')
max_label_id = np.max(cluster_labels)
print('There are %d clusters'%max_label_id)
for i in range(1, max_label_id + 1):
    members = df_norm.index[cluster_labels == i]
    print('Cluster %d has %d customers: %s'%(
            i, len(members), ' '.join(map(str, members))))


# In[9]:

# (1e continued)
# Evaluate the clustering result
silh_score = skm.silhouette_score(df_norm, cluster_labels)
print ('Silhouette coefficient: %f'%silh_score)


# In[10]:

# 1f. k-means clustering

# Build PCA model for visualizing high-dim cluster results
pca_model = skd.PCA(n_components=2).fit(df_norm)
pca_data = pca_model.transform(df_norm)
df_pca = pd.DataFrame(pca_data, columns=['pca0', 'pca1'])

for k in range(2,6):
    kmeans_model = skc.KMeans(k).fit(df_norm)
    centroids = kmeans_model.cluster_centers_
    centroids_pca = pca_model.transform(centroids)
    cluster_labels = kmeans_model.labels_
    df_pca['label'] = cluster_labels
    silh_score = skm.silhouette_score(df_norm, cluster_labels)
    
    plt.figure()
    f = sns.lmplot(x='pca0', y='pca1', data=df_pca, 
                   hue='label', fit_reg=False, 
                   scatter_kws={'s': 100})
    f.ax.scatter(centroids_pca[:,0], centroids_pca[:,1], 
                 marker='+', s=200, linewidths=2, color='black')
    plt.title('k-means: k=%d\nSilhouette_score=%f'%(
                k, silh_score), fontsize=15)


# In[11]:

# (1f continued) 
# Determine the best number of clusters: Elbow method
ks = range(1, 11)
kmeans_models = [skc.KMeans(k).fit(df_norm) for k in ks]
centroids = [m.cluster_centers_ for m in kmeans_models]
D_k = [spd.cdist(df_norm, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D, axis=1) for D in D_k]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d) / df_norm.shape[0] for d in dist]

plt.plot(ks, avgWithinSS, 'b*-')
plt.xlabel('Number of clusters')
plt.ylabel('Average within-group sum of squares')
plt.title('Elbow Apporach for k-Means Clustering')

# add text annotation
elbowIdx = 2  # manually picked elbow point
arrow_end = (ks[elbowIdx],avgWithinSS[elbowIdx])
arrow_start = (arrow_end[0] + 1, arrow_end[1] + 0.5)
plt.gca().annotate('The "Elbow" Point (k=3)', 
                   xy=arrow_end, 
                   xytext=arrow_start,
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   fontsize=15)


# It can be seen quite obviously that there is an angle (hence an elbow) at k=3.

# In[12]:

# (1f continued)
# Determine the best number of clusters: Information Criterion approach
ks = range(1, 11)
gmms = [skmix.GMM(k).fit(df_norm) for k in ks]
bics = [g.bic(df_norm) for g in gmms]
plt.plot(ks, bics, 'b*-')

kIdx = np.argmin(bics)  # <-- the selected index
plt.plot(ks[kIdx], bics[kIdx], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r',
         markerfacecolor='None')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.title('BIC for GMM')


# The BIC (Bayesian Information Criterion) approach selects k=8 to be the best number of clusters. Although I would probably select 4 given this graph, as the BIC doesn't drop much after that point.

# In[13]:

# 1g. Evaluation with ground-truth
ground_truth = [[170, 198, 239, 384],
                [123, 166, 268, 332],
                [154, 222, 293, 334, 335, 537, 584]]

df_truth = pd.DataFrame(
            [(x,i) for i,c in enumerate(ground_truth) for x in c], 
            columns=['CustomerID','GroundTruth'])
df_truth.set_index('CustomerID', inplace=True)
df_norm['GroundTruth'] = df_truth['GroundTruth']
true_labels = df_norm['GroundTruth']

cluster_labels = skc.KMeans(3).fit(df_norm).labels_
skm.adjusted_rand_score(true_labels, cluster_labels)


# In[14]:

skm.normalized_mutual_info_score(true_labels, cluster_labels)


# Both RI and NMI give 1.0, which indicates we obtain a perfect clustering compared to the ground truth.

# Solutions to Questions 2 and 3 omitted.
