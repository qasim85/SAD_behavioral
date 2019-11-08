#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 00:00:08 2019

@author: qasimbukhari
"""


#import sys
#sys.modules[__name__].__dict__.clear()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
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

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")




#ds = pd.read_csv('/Users/qasimbukhari/Documents/ForSatra/combined_numeric_May2019_without1stcol.csv', na_values=[999, '', 'nan'])
ds = pd.read_csv('/Users/qasimbukhari/Documents/ForSatra/combined_numeric_Aug2019_3.csv', na_values=[999, '', 'nan'])

ds=ds.drop(['ID', 'extra1', 'DOB', 'LSAStotal_baseline'], axis=1)

ds=ds.dropna(axis=1)


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


#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
 



allresults = {}

#nfolds=500
#
#
#from sklearn import svm, datasets
#from sklearn.model_selection import GridSearchCV
#iris = datasets.load_iris()
#parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
#svc = svm.SVC(gamma="scale")
#clf = GridSearchCV(svc, parameters, cv=5)
#clf.fit(iris.data, iris.target)
#print clf.best_estimator_
#print("%s" % (clf.best_params_))





#    # Support Vector Machines
#
#svc = SVC(max_iter=100000)
#
#sss = StratifiedShuffleSplit(n_splits=20, test_size=0.33333, random_state=5195)
#
#grid_search_CV = GridSearchCV(svc, parameters, cv=sss, scoring=accuracy_scorer)
#grid_search_CV.fit(X, Y)
#print("score equals %f" % (grid_search_CV.best_score_))
#print("%s" % (grid_search_CV.best_params_))
#svc = grid_search_CV.best_estimator_






#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
#def svc_param_selection(X, y, nfolds):
#    Cs = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_


#parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
#svc = svm.SVR(gamma="scale")
#clf1 = GridSearchCV(svc, parameters, cv=5)


from sklearn.linear_model import Ridge

RD_model = Ridge()
parameters = {'alpha': list(x / 10 for x in range(0, 101)),
                    'fit_intercept': [True,False], 
                    'normalize' :[False, True],
                    'solver': ['lsqr', 'auto', 'svd']}
clf1 = GridSearchCV(RD_model, parameters, cv = 10)





clf1.fit(X[:, 0:132], Y[:,12])
print clf1.best_estimator_
print("%s" % (clf1.best_params_))

clf = clf1.best_estimator_

    
for idx in range(Y.shape[1]):
    results = [[], []]
    scores = []
    clfs = []
    sss = ShuffleSplit(n_splits=500, test_size=0.2, random_state=0)
    #for train, test in sss.split(np.zeros(Y.shape[0]), Y[:,idx]): 
    for train, test in sss.split(X):   
    ##for train, test in ShuffleSplit(Y.shape[0], n_iter=200, test_size=0.2):
    #for train, test in ShuffleSplit(n_splits=200, random_state=0, test_size=0.2, train_size=None):
    #for train, test in ShuffleSplit(n_splits=200, random_state=0, test_size=0.2, train_size=None):   
        #clf1 = ExtraTreesRegressor(n_estimators=10, max_depth=None, min_samples_split=2)

        #clf = SVR(kernel='rbf')
        #clf = RandomForestRegressor(n_estimators= 1000, max_depth=30, random_state=0) 

#        svr_lin = SVR(kernel='linear')
#        ridge = Ridge(random_state=1)
#        svr_rbf = SVR(kernel='rbf')
#        rf = RandomForestRegressor(n_estimators= 10, max_depth=30, random_state=0)    
    
    
    


#        parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
#        svc = svm.SVC(gamma="scale")
#        clf = GridSearchCV(svc, parameters, cv=2)
#        clf.fit(X[train], Y[train, idx])
#        print clf.best_estimator_
#        print("%s" % (clf.best_params_))

    
    
        #parameters = {'min_samples_split': [1, 5, 10, 50],
        #              'max_features': [0.1, 0.5, 0.9]}
        #clf1 = GridSearchCV(clf1, param_grid=parameters)
#        clf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#            ("standardize", StandardScaler()),
#            ("forest", clf1)])
    
    
    
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
        print "Ridge model:", pretty_print_linear(clf.coef_)
  
        results[0].extend(Y[test, idx])
        results[1].extend(ytest)
        num=num+1
        print num
        scores.append(explained_variance_score(Y[test, idx], ytest))
        #clfs.append(clf.steps[-1][1].best_estimator_.feature_importances_)
        clfs.append(clf.coef_)
        
        #rmse_ridge = np.sqrt(clf.fit(X[train], Y[train, idx]))
        #rmse_ridge = np.sqrt(-cross_val_score(Ridge(), X[train], Y[train, idx], scoring="neg_mean_squared_error", cv = 5))
        #print rmse_ridge
        
    score = explained_variance_score(results[0], results[1])
    allresults[lsas_keys[idx]] = {'results': results, 'r2': score,
                               'scores': scores, 'median_r2': np.median(scores),
                               'clfs': clfs}




f1_results = pd.DataFrame([{'name': k, 
                           'median_R2': allresults[k]['median_r2'],
                           'r2': allresults[k]['r2']} 
                          for k in sorted(allresults)]).sort_values('median_R2', ascending=False)
#f1_results.index = f1_results.name
display(pd.DataFrame(f1_results['median_R2']))


#np.median(allresults[k]['scores'])



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


## 1. Ridge 2. RF, 3 SVM, 4 ET
##sdf2=np.array([[ 4.27347466, 3.96585711,3.34552716, 1.05784079,4.65615188,8.56953322,17.80777335,21.25535053,17.77228658,22.0980826,
##       14.57691378, 25.65484619,29.8768224 ,25.3693599 ,23.2180457 ,20.47141291], 
#
#
#
#sdf2=np.array([
#        
## Ridge        
#        [0.04273475, 0.03965857, 0.03345527, 0.01057841, 0.04656152,
#        0.08569533, 0.17807773, 0.21255351, 0.17772287, 0.02098083,
#        0.14576914, 0.25654846, 0.29876822, 0.2536936 , 0.23218046,
#        0.20471413],
#
#
## RF
#        [0.001809 , 0.006870, 0.015480, 0.045515,    # 0, 1,2,3, 
#        0.025960, 0.092036, 0.193645, 0.213794, 0.181180  ,0.257111   ,0.168614  ,0.211746  ,       # 4, 5,6,7,8,9,10,11
#       0.289989, 0.226086,  0.161541,  0.137310] ,       # 12, 13,17,25,
#
#
## SVM
#        [0.007111, 0.045868 , 0.012551 , 0.061792 ,    # 0, 1,2,3, 
#        0.014886, 0.094043, 0.140824, 0.154616, 0.160451, 0.186842, 0.117982, 0.241062   ,       # 4, 5,6,7,8,9,10,11
#       0.301237, 0.262650, 0.222714, 0.205427],       # 12, 13,17,25,
#
#
#
#
## ET
#        [ -0.105656, -0.222316, -0.108070, -0.163439,    # 0, 1,2,3, 
#        -0.149938, 0.029457, 0.121139 ,0.135219, 0.118195, 0.185345 , 0.126904, 0.209404 ,       # 4, 5,6,7,8,9,10,11
#       0.250366, 0.191952, 0.145078, 0.086946 ]       # 12, 13,17,25,
#           ])    







#sdf3=np.transpose(sdf2)


#pp=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 25, 37]
sdf.index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 25, 37]

plt.plot(sdf.index, sdf.values)
#plt.plot(sdf.index, sdf.values)
#plt.legend(labels)
#ph = sdf.plot(x=range(1,16))
plt.ylabel('Explained variance (R^2)')
plt.xlabel('weeks')



#ph = sdf.plot(x=positions[1:])
#ylabel('Explained variance (R^2)')
#xlabel('weeks')

#positions = np.array([int(val.split('_')[1]) for val in dfconcat.keys()])
#xth = xticks(positions[1:], sdf.index, rotation=90)





#for key in f1_results.name:
#    if f1_results.loc[f1_results.name == key, 'median_R2'].values > 0:
#        feature_importance = np.median(np.array(allresults[key]['clfs']), axis=0)
#        indices = np.argsort(feature_importance)[-20:][::-1]
#        print key
#        print np.array(features)[indices]



for key in f1_results.name:
    if f1_results.loc[f1_results.name == key, 'median_R2'].values > 0:
        feature_importance = np.median(np.array(allresults[key]['clfs']), axis=0)
        indices = np.argsort(feature_importance)[-20:][::-1]
        print key
        print np.array(features)[indices]
        print np.array(feature_importance)[indices]






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
print np.array(FinalFeatures)[FinalIndices]
print np.array(features)[FinalIndices]

