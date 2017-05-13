import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

def predict_missing_variable(df, variable):
    small_df = df[df[variable] != np.nan]

    train_y = small_df[variable]
    train_x = small_df.drop(variable, axis=1)
    test_x = df[df[variable] == np.nan].drop(variable,axis=1)

    print(train_x.head())

    pred = runXgb(train_x,train_y,test_x)
    print(len(df[df[variable] == np.nan]))
    df.loc[df[variable] == np.nan, variable] = pred
    print(len(df[df[variable] == np.nan]))

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
