import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

def predict_missing_variable(df, variable):
    
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))
            
    df[variable].fillna(-1,inplace=True)
    
    small_df = df[df[variable] != -1]

    train_y = small_df[variable]
    train_x = small_df.drop(variable, axis=1)
    test_x = df[df[variable] == -1].drop(variable,axis=1)

    pred = runXgb(train_x,train_y,test_x)
    
    df.loc[df[variable] == -1, variable] = pred	 
    return df[variable]
    
def runXgb(train_x, train_y, test_x = None):

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(train_x, train_y)


    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
        verbose_eval=50)

    num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds)

    if test_x is not None:
        dtest = xgb.DMatrix(test_x)
        return model.predict(dtest)
