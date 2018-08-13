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
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

pd.options.display.width=2000

def create_train_batch(train_event='event000001000', folder = '../input/train_sample/train_100_events'):

    def add_features(df):
        # Add features
        df.loc[:, 'r'] = np.sqrt(df.x ** 2 + df.y ** 2)
        df.loc[:, 'd'] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
        df.loc[:, 'theta'] = np.arctan2(df.y, df.x)
        df.loc[:, 'cos_theta'] = np.cos(df.loc[:, 'theta'])
        df.loc[:, 'sin_theta'] = np.sin(df.loc[:, 'theta'])
        df.loc[:, 'phi'] = np.arctan2(df.z, df.r)
        df.loc[:, 'cos_phi'] = np.cos(df.loc[:, 'phi'])
        df.loc[:, 'sin_phi'] = np.sin(df.loc[:, 'phi'])


    hits, cells, particles, truth = load_event("%s/%s"%(folder,train_event))

    cells_m = cells.groupby(by='hit_id').agg(np.average)
    cells_m.loc[:, 'ch_r'] = np.sqrt(cells_m.ch0 ** 2 + cells_m.ch1 ** 2)
    cells_m.loc[:, 'ch_theta'] = np.arctan2(cells_m.ch1, cells_m.ch0)

    merged = truth.merge(hits, on='hit_id')
    merged = merged.merge(cells_m, left_on='hit_id', right_index=True)

    merged = merged.set_index('hit_id')
    merged = merged.loc[merged.particle_id != 0]

    # g_10 = merged.groupby(by='particle_id').count().sort_values(by='x',ascending=False).head(100)
    g_length = merged.groupby(by='particle_id').count().where(cond=lambda df: df.x >=10).dropna()
    df = merged.loc[merged.particle_id.isin(g_length.index)]
    del g_length

    add_features(df)
    df = df.sort_values(by=['particle_id', 'd'])

    return df

def createXY(df,return_map = False,MAX_NUM_TRACKS=20):
    p_map = {}
    for p, idx in df.groupby(by='particle_id').groups.items():
        v = df.loc[idx]
        v = v.loc[:, FEATURE_COLUMNS]
        v = v.as_matrix()
        v = x_scaler.transform(v)
        v = v[:-1]
        v1 = np.zeros(shape=(MAX_NUM_TRACKS, v.shape[1]))
        v1[:v.shape[0]] = v[:]
        p_map[p] = v1

    y_map = {}
    for p, idx in df.groupby(by='particle_id').groups.items():
        v = df.loc[idx]
        v = v.loc[:, ['x', 'y', 'z']]
        v = v.as_matrix()
        v = y_scaler.transform(v)
        v = v[1:]
        v1 = np.zeros(shape=(MAX_NUM_TRACKS, v.shape[1]))
        v1[:v.shape[0]] = v[:]
        y_map[p] = v1

    X = np.zeros((len(p_map), MAX_NUM_TRACKS, len(FEATURE_COLUMNS)))
    for i, item in enumerate(p_map.values()):
        X[i] = item

    Y = np.zeros((len(y_map), MAX_NUM_TRACKS, 3))
    for i, item in enumerate(y_map.values()):
        Y[i] = item

    if return_map:
        return X,Y,p_map,y_map
    else :
        return X, Y


def mymodel(num_trk,num_f,n_a=20):

    input_X = KL.Input(shape=(num_trk,num_f))
    m = KL.Masking(mask_value=0)(input_X)

    m = KL.TimeDistributed(KL.Dense(16))(m)
    m = KL.TimeDistributed(KL.Dense(16))(m)
    m = KL.TimeDistributed(KL.Dense(16))(m)
    m = KL.LSTM(n_a,return_sequences=True)(m)
    m = KL.TimeDistributed(KL.Dropout(0.2))(m)
    m = KL.LSTM(n_a, return_sequences=True)(m)
    #m = KL.TimeDistributed(KL.Reshape((-1,1)))(m)
    #m = KL.TimeDistributed(KL.Conv1D(8,2,activation='relu'))(m)
    #m = KL.TimeDistributed(KL.GlobalAveragePooling1D())(m)
    m = KL.TimeDistributed(KL.Dropout(0.2))(m)
    m = KL.TimeDistributed(KL.Dense(3))(m)

    model = models.Model([input_X],m)
    return model

MAX_NUM_TRACKS = 20
FEATURE_COLUMNS = ['x', 'y', 'z', 'r', 'd', 'theta','cos_theta','sin_theta', 'phi','cos_phi','sin_phi','ch0','ch1','ch_r','ch_theta','value']

train_df = create_train_batch(train_event='event000001000')
val_df = create_train_batch(train_event='event000001001')

c_df = np.vstack((train_df.loc[:,FEATURE_COLUMNS].as_matrix(),val_df.loc[:,FEATURE_COLUMNS].as_matrix()))

x_scaler = pp.StandardScaler()
x_scaler.fit(c_df)
y_scaler = pp.StandardScaler()
y_scaler.fit(c_df[:,0:3])

X, Y = createXY(train_df)
X_val, Y_val = createXY(val_df)

model = mymodel(MAX_NUM_TRACKS,len(FEATURE_COLUMNS),n_a)
adam = optimizers.Adam(lr=0.01)
model.compile(optimizer=adam,loss='mse',metrics=['mse'])
model.fit(X,Y,epochs=10,validation_data=(X_val,Y_val),verbose=2)

### Prediction

points_xyz = x_scaler.transform(val_df.loc[:,FEATURE_COLUMNS])
nearest = NearestNeighbors(n_neighbors=1)
nearest.fit(points_xyz[:,0:3])

def predict(starting_point,model,nearest,num_of_preds=MAX_NUM_TRACKS):
    X_l= np.zeros((1,MAX_NUM_TRACKS,len(FEATURE_COLUMNS)))
    P_l = np.zeros((MAX_NUM_TRACKS,3))

    nearest_point = starting_point
    for i in range(0,num_of_preds):
        X_l[0,i] = nearest_point
        p = model.predict(X_l)
        distances,nn_idx = nearest.kneighbors([p[0,i]],n_neighbors=10,return_distance=True)
        print("Step %s for predicted next point: %s"%(i,p[0,i]))
        for npoint ,distance in zip(points_xyz[nn_idx[0]],distances[0]):
            print("[%s]%s"%(distance,npoint[0:3]))

        nearest_point = points_xyz[nn_idx[0,0]]
        P_l[i] = nearest_point[0:3]
    return P_l
