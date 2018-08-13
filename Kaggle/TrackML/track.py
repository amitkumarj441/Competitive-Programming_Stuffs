import numpy as np
import pandas as pd
from trackml.dataset import load_event
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as pp
from sklearn.cluster import DBSCAN
import sklearn.metrics as metrics
from sklearn.metrics import pairwise
import keras
import keras.backend as K
from keras import layers as KL,models,optimizers


pd.options.display.width=2000

def create_features(df):
    df.loc[:, 'r'] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, 'd'] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df.loc[:, 'theta'] = np.arctan2(df.y, df.x)
    df.loc[:, 'cos_theta'] = np.cos(df.loc[:, 'theta'])
    df.loc[:, 'sin_theta'] = np.sin(df.loc[:, 'theta'])
    df.loc[:, 'phi'] = np.arctan2(df.z, df.r)
    df.loc[:, 'cos_phi'] = np.cos(df.loc[:, 'phi'])
    df.loc[:, 'sin_phi'] = np.sin(df.loc[:, 'phi'])


hits, cells, particles, truth = load_event('../input/train_sample/train_100_events/event000001000')

merged = truth.merge(hits,on='hit_id')
merged = merged.set_index('hit_id')
merged = merged.loc[merged.particle_id !=0]
create_features(merged)

v18_l12 = merged.loc[ (merged.volume_id==18) & (merged.layer_id==12)]

cells_l12 =  cells.loc[cells.hit_id.isin(v18_l12.index)]

t =cells.loc[cells.hit_id.isin(v18_l12.index),['ch0','ch1']].agg([np.max,np.min])
ch0_max , ch0_min = t['ch0']
ch1_max , ch1_min = t['ch1']

hit_ids = cells_l12.hit_id.unique()

X_ch = np.zeros((len(hit_ids),ch0_max+1,ch1_max+1))
hit_id_img={}
for i, hi in enumerate(hit_ids):
    hit_id_img[hi] = np.zeros((ch0_max + 1,ch1_max + 1))
    c = cells.loc[cells.hit_id == hi]
    hit_id_img[hi][c.ch0.values,c.ch1.values] = c.value
    X_ch[i] = hit_id_img[hi]

FEATURE_COLUMNS = ['x', 'y', 'z', 'r', 'd', 'theta','cos_theta','sin_theta', 'phi','cos_phi','sin_phi']

x_scaler = pp.StandardScaler()
X_p = x_scaler.fit_transform(v18_l12.loc[:,FEATURE_COLUMNS])
