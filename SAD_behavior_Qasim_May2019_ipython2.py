#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:35:33 2019

@author: qasimbukhari
"""
#import sys
#sys.modules[__name__].__dict__.clear()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.linear_model import RandomizedLasso, Lasso, LassoCV, LassoLarsCV, RidgeCV, LinearRegression
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, r2_score, f1_score, matthews_corrcoef
from IPython.display import display
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


#read reducedset.csv as ds
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")




#ds = pd.read_csv('/Users/qasimbukhari/Documents/ForSatra/combined_numeric_May2019_without1stcol.csv', na_values=[999, '', 'nan'])
ds = pd.read_csv('/Users/qasimbukhari/Documents/ForSatra/combined_numeric_Aug2019_2.csv', na_values=[999, '', 'nan'])

ds=ds.drop(['ID', 'LSAStotal_baseline'], axis=1)

#features = [key for key in ds.keys() if not key.startswith('LSAS_') | key.startswith('Site_')]

features = [key for key in ds.keys() if not key.startswith('LSAS_')]
lsas_keys = [key for key in ds.keys() if key.startswith('LSAS_')]




#site_keys = [key for key in ds.keys() if key.startswith('Site_')]
#Site_x_SMU
print lsas_keys
print features
subjs = ds.loc[:, lsas_keys].dropna().index
#subjs = subjs.ix[:, site_keys].dropna().index

X = ds.loc[subjs, features].values




Y_total = ds.loc[subjs, lsas_keys].values  # LSAS  total

Y_change = ds.loc[subjs, lsas_keys[1:]].values - ds.loc[subjs, lsas_keys[0]].values[:, None] # LSAS change



Y12 = ds.loc[subjs, 'LSAS_12'].values
print Y12
Y13 = ds.loc[subjs, 'LSAS_13'].values
print Y13
Y37 = ds.loc[subjs, 'LSAS_37'].values
print Y37

Y37change=Y_change[:,15]
Y12change=Y_change[:,11]
Y13change=Y_change[:,12]


## Y values
Y=Y_change



num=0
allresults = {}

from sklearn.metrics import explained_variance_score, r2_score, f1_score, matthews_corrcoef

def null_variance_score(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    ynull = []
    for train, test in LeaveOneOut(len(y_true)):
        ynull.append(np.average(y_true[train]))
    return 1 - np.sum((y_true - y_predict)**2)/np.sum((np.array(ynull) - y_true)**2)
#explained_variance_score = null_variance_score



allresults = {}

    
for idx in range(Y.shape[1]):
    results = [[], []]
    scores = []
    clfs = []
    sss = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    #for train, test in sss.split(np.zeros(Y.shape[0]), Y[:,idx]): 
    for train, test in sss.split(X):   
    ##for train, test in ShuffleSplit(Y.shape[0], n_iter=200, test_size=0.2):
    #for train, test in ShuffleSplit(n_splits=200, random_state=0, test_size=0.2, train_size=None):
    #for train, test in ShuffleSplit(n_splits=200, random_state=0, test_size=0.2, train_size=None):   
        clf1 = ExtraTreesRegressor(n_estimators=1000, max_depth=None, min_samples_split=2)

    
    

        svr_lin = SVR(kernel='linear')
        ridge = Ridge(random_state=1)
        svr_rbf = SVR(kernel='rbf')
        rf = RandomForestRegressor(n_estimators= 10, max_depth=30, random_state=0)    
    
    
    
    
    
    
        parameters = {'min_samples_split': [1, 5, 10, 50],
                      'max_features': [0.1, 0.5, 0.9]}
        #clf1 = GridSearchCV(clf1, param_grid=parameters)
        clf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("standardize", StandardScaler()),
            ("forest", clf1)])
    
    
    
#    from sklearn.c import RandomizedSearchCV
## Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
## Number of features to consider at every split
#max_features = ['auto', 'sqrt']
## Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
## Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
## Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
## Method of selecting samples for training each tree
#bootstrap = [True, False]
## Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}
#pprint(random_grid)
    
    
    
    
    
    
        ytest = clf.fit(X[train], Y[train, idx]).predict(X[test])
        results[0].extend(Y[test, idx])
        results[1].extend(ytest)
        num=num+1
        print num
        scores.append(explained_variance_score(Y[test, idx], ytest))
        #clfs.append(clf.steps[-1][1].best_estimator_.feature_importances_)
        clfs.append(clf.steps[-1][1].feature_importances_)

    score = explained_variance_score(results[0], results[1])
    allresults[lsas_keys[idx]] = {'results': results, 'r2': score,
                               'scores': scores, 'median_r2': np.median(scores),
                               'clfs': clfs}




f1_results = pd.DataFrame([{'name': k, 
                           'median_R2': allresults[k]['median_r2'],
                           'r2': allresults[k]['r2']} 
                          for k in sorted(allresults)]).sort_values('median_R2', ascending=False)
f1_results.index = f1_results.name
display(pd.DataFrame(f1_results['median_R2']))

    
    



#import numpy as np
#fh = plt.figure(figsize=(15,5))
#sdf = pd.DataFrame(f1_results['median_R2']).sort_index()*100
#
#plt.plot(sdf.index, sdf.values)
#
##ph = sdf.plot(x=range(1,16))
#plt.ylabel('Explained variance (R^2)')
#plt.xlabel('weeks')




import numpy as np
fh = plt.figure(figsize=(15,5))
sdf = pd.DataFrame(f1_results['median_R2']).sort_index()*100


pp=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 25, 37]


plt.plot(sdf.index, sdf.values)

#ph = sdf.plot(x=range(1,16))
plt.ylabel('Explained variance (R^2)')
plt.xlabel('weeks')



#ph = sdf.plot(x=positions[1:])
#ylabel('Explained variance (R^2)')
#xlabel('weeks')

#positions = np.array([int(val.split('_')[1]) for val in dfconcat.keys()])
#xth = xticks(positions[1:], sdf.index, rotation=90)





for key in f1_results.name:
    if f1_results.loc[f1_results.name == key, 'median_R2'].values > 0:
        feature_importance = np.median(np.array(allresults[key]['clfs']), axis=0)
        indices = np.argsort(feature_importance)[-20:][::-1]
        print key
        print np.array(features)[indices]



feature_importance2=[]
indices2=[]
for key in f1_results.name[0:5]:
    if f1_results.loc[f1_results.name == key, 'median_R2'].values > 0:
        feature_importance = np.median(np.array(allresults[key]['clfs']), axis=0)
        feature_importance2=np.concatenate((feature_importance, feature_importance2), axis=0)
        indices = np.argsort(feature_importance)[-50:][::-1]
        indices2 = np.concatenate((indices, indices2), axis=0)
        print key
        print np.array(features)[indices]
        print np.array(feature_importance)[indices]



FinalFeatures=numpy.sum(feature_importance2.reshape((5, -1)), axis = 0)
FinalIndices = np.argsort(FinalFeatures)[-20:][::-1]
print np.array(FinalFeatures)[FinalIndices]  #  Median
MedianImpFeatures=np.array(FinalFeatures)[FinalIndices] 
print np.array(numpy.std(feature_importance2.reshape((5, -1)), axis = 0))[FinalIndices]  # Standard Deviation
StdImpFeatures = np.array(numpy.std(feature_importance2.reshape((5, -1)), axis = 0))[FinalIndices] 
print np.array(features)[FinalIndices]



import matplotlib.pyplot as plt



ind = np.arange(MedianImpFeatures.shape[0])    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, MedianImpFeatures, width, yerr=StdImpFeatures)
plt.ylabel('Median Feature Importance Values')
plt.title('Feature  Importance')
plt.gcf().subplots_adjust(bottom=0.25)
plt.xticks(ind, (np.array(features)[FinalIndices]), rotation='vertical')
plt.show()




