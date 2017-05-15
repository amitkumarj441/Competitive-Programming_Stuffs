import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from Pred_na import *

train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

def distribucion_price(df, variable):
    
    df[variable].dropna(inplace = True)
    price = df[variable].values
    
  	sns.distplot(price,)
    plt.show()
    
distribucion_price(train_df, "life_sq")
    
