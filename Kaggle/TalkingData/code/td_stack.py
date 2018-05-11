import numpy as np 
import pandas as pd 
matplotlib.pyplot as plt 
import gc 
from time import time 
from contextlib import contextmanager 

# timer function 
@contextmanager 
def timer(operationName):
	t0 = time()
	yield 
	print("{} uses {:.1f} seconds".format(operationName, time()-t0))

#read data 
path = '~/Downloads/TalkingData'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
	'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}

with timer('read training data'):
	train = pd.read_csv(path+'train.csv', 
		usecols=train_cols, 
		dtype=dtypes,
		parse_dates=['click_time'])

with timer('read test data'):
	test = pd.read_csv(path+'test.csv',
		usecols=test_cols,
		dtype=dtypes,
		parse_dates=['click_time'])

with timer('read submission file'):
    submission = pd.read_csv(path+'sample_submission.csv')

# extract 'day' and 'hour' from feature 'click_time' 
train['day'] = train['click_time'].dt.day.astype('uint8')
train['hour'] = train['click_time'].dt.hour.astype('uint8')

test['day'] = test['click_time'].dt.day.astype('uint8')
test['hour'] = test['click_time'].dt.hour.astype('uint8')

# create new feature by groupby and aggregation 
groupby_list = [
	#variance per combination of features
	{'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
	{'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
	{'groupby': ['ip','day','channel'], 'select': 'hour', 'agg', 'var'},
	
	#count per combination of features
	{'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
	{'groupby': ['ip','app'], 'select': 'channel', 'agg': 'count'},
	{'groupby': ['ip','app','os'], 'select': 'channel', 'agg': 'count'},
	{'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'}
	
	# mean per combination of feature
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # average clicks by distinct users for each app
    {'groupby': ['ip'], 'select': 'ip', 'agg': lambda x: len(x)/len(x.unique()),
    'agg_name': 'avgViewDist'},
    
    # unique counts per ip and combination of features
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'nunique'}, 
    
    #how many cumulative count of app per ip and other features
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}
	]

with timer('create groupby features'):
    for item in groupby_list:
	    name = item['agg_name'] if 'agg_name' in item else item['agg']
	    feature_name = '{}_{}_{}'.format('_'.join(item['groupby']), name, item['select'])
	
	    used_features = list(set(item['groupby'] + item['select']))
	
	    temp = train[used_features].\
	        groupby(item['groupby'])[item['select']].\
	        agg(item['agg']).\
	        reset_index().\
	        rename(index=str, columns={item['select']: feature_name})
	
	    temp1 = test[used_features].\
	        groupby(item['groupby'])[item['select']].\
	        agg(item['agg']).\
	        reset_index().\
	        rename(index=str, columns={item['select']: feature_name})
	
	    if item['agg'] == 'cumcount':
		    train[feature_name] = temp[0].values
		    test[feature_name] = temp1[0].values
	    else:
		    train = train.merge(temp, on=item['groupby'], how='left')
		    test = test.merge(temp1, on=item['groupby'], how='left')

	    del temp, temp1
	    gc.collect()
        
# time it takes for a next click?
with timer('create next_click feature'):
	train['next_click'] = train[['ip','os','device','app','click_time']].\
		groupby(['ip','os','device','app']).\
		click_time.\
		transform(lambda x: x.diff().shift(-1)).\
		dt.seconds

	test['next_click'] = train[['ip','os','device','app','click_time']].\
		groupby(['ip','os','device','app']).\
		click_time.\
		transform(lambda x: x.diff().shift(-1)).\
		dt.seconds

train.drop('click_time', axis=1, inplace=True)
test.drop('click_time', axis=1, inplace=True)

gc.collect()  

#train a neural net classifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

# define features and target 
features = ['app','device','os','channel','hour','dayofweek']
target = ['is_attributed']

"""
candidate one based on 5-fold CV:
MLPClassifier(hidden_layer_sizes=(150,),
             activation='logistic',
             alpha=2.8e-6,
             learning_rate_init=2.024e-3,
             shuffle=False,
             random_state=0)
"""

#train lightGBM classifier 
import lightgbm as lgb 
from lightgbm.sklearn import LGBMClassifier 

"""
candidate one based on 5-fold CV:
LGBMClassifier(boosting_type='gbdt',
              num_leaves=65,
              max_depth=6,
              learning_rate=0.2,
              n_estimators=1000,
              is_unbalance=True,
              min_child_weight=350,
              min_child_samples=100,
              subsample=0.6,
              colsample_bytree=0.6,
              n_jobs=8,
              random_state=0)
"""

# train xgboost Classifier 
import xgboost as xgb 
from xgboost.sklearn import XGBClassifier

"""
candidate one based on 5-fold CV:
XGBClassifier(max_depth=6,
             learning_rate=0.2,
             n_estimators=500,
             silent=False,
             objective='binary:logistic',
             n_jobs=8,
             min_child_weight=3,
             subsample=0.6,
             colsample_bytree=0.6,
             scale_pos_weight=350,
             random_state=0)
"""

#make meta-learner based on lgb,xgboost and simple nn
X = train[features].values
y = train[target].values.flatten()

kFold = StratifiedKFold(n_splits = 5, shuffle = False, random_state = 666)
kf = kFold.split(X, y)

clfs = [
    MLPClassifier(hidden_layer_sizes=(150,),
             activation='logistic',
             alpha=2.8e-6,
             learning_rate_init=2.024e-3,
             shuffle=False,
             random_state=0),
    LGBMClassifier(boosting_type='gbdt',
              num_leaves=65,
              max_depth=6,
              learning_rate=0.2,
              n_estimators=1000,
              is_unbalance=True,
              min_child_weight=350,
              min_child_samples=100,
              subsample=0.6,
              colsample_bytree=0.6,
              n_jobs=8,
              random_state=0),
    XGBClassifier(max_depth=6,
             learning_rate=0.2,
             n_estimators=500,
             silent=False,
             objective='binary:logistic',
             n_jobs=8,
             min_child_weight=3,
             subsample=0.6,
             colsample_bytree=0.6,
             scale_pos_weight=350,
             random_state=0)
]

blend_train = np.zeros((train.shape[0], len(clfs)))
blend_test = np.zeros((submission.shape[0], len(clfs)))

for i, clf in enumerate(clfs):
    print("{},{}".format(i+1, clf))
    blend_test_i = np.zeros((submission.shape[0], 5))
    for j, (train_fold, test_fold) in enumerate(kf):
        print("Fold {}".format(j+1))
        X_train = X[train_fold]
        y_train = y[train_fold]
        X_test = X[test_fold]
        y_test = y[test_fold]
        
        clf.fit(X_train, y_train)
        
        print("Prediction for fold {}, clf {}".format(j+1, i+1))
        y_pred = clf.predict_proba(X_test)[:,1]
        blend_train[test_fold, i] = y_pred
        
        print("Prediction for test set, fold {}, clf {}".format(j+1, i+1))
        blend_test_i[:, j] = clf.predict_proba(test.values)[:,1]
    
    blend_test[:,i] = blend_test_i.mean(axis=1)
    print("Finish clf {}".format(i+1))
print("Finish all stacking.")

# logistic regression as 2nd stage learner with 1st stage prediction as derived features 
lrc = LogisticRegression()
lrc.fit(blend_train, y)
y_submission = lrc.predict_proba(blend_test)[:, 1]

# try stretching the predictions
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

submission['is_attributed'] = y_submission

submission.to_csv('stack_sub.csv', index=False)
