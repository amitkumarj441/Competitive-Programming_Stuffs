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

train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

def distribucion_price(df):

    price = df["price_doc"].values
    print(price)
    sns.distplot(price)
    plt.show()

vr = []

for x in train_df.columns:
    if train_df[x].dtypes == 'float':
        vr.append((np.var(train_df[x]),x))

vr.sort()
print(vr[:20])
print(vr[-20:])
