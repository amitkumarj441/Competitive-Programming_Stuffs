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

hits, cells, particles, truth = load_event('../input/train_sample/train_100_events/event000001000')

merged = truth.merge(hits,on='hit_id')
merged = merged.set_index('hit_id')
merged = merged.loc[merged.particle_id !=0]

s_hits =cells.groupby(by='hit_id')['hit_id'].count() ==1

merged = merged.loc[s_hits].merge(cells,right_on='hit_id',left_index=True)

merged.loc[:,['volume_id','layer_id','module_id','ch0','ch1', 'x','y','z']]


X = [ merged.loc[:,'volume_id'], merged.loc[:,'layer_id'],merged.loc[:,'module_id'],merged.loc[:,'ch0'],merged.loc[:,'ch1'] ]

Y = merged.loc[:,['x','y','z']]

v_input = KL.Input(shape=(1,))
v_input_emb = KL.Embedding(np.max(merged.volume_id.values)+1,5)(v_input)

l_input = KL.Input(shape=(1,))
l_input_emb = KL.Embedding(np.max(merged.layer_id.values)+1,5)(l_input)

m_input = KL.Input(shape=(1,))
m_input_emb = KL.Embedding(np.max(merged.module_id.values)+1,5)(m_input)

ch0_input = KL.Input(shape=(1,))
ch1_input = KL.Input(shape=(1,))

x = KL.Concatenate()([v_input_emb,l_input_emb,m_input_emb])
x = KL.Lambda(lambda a: K.squeeze(a,1))(x)
c = KL.Concatenate()([ch0_input,ch1_input])
x = KL.Concatenate()([c,x])
x= KL.Dropout(0.3)(x)
x = KL.Dense(8,activation='relu')(x)
x= KL.Dropout(0.3)(x)
c = KL.Dense(8,activation='relu')(c)
x = KL.BatchNormalization()(x)
x= KL.Dropout(0.3)(x)
x = KL.Dense(8,activation='relu')(x)
x = KL.BatchNormalization()(x)
x= KL.Dropout(0.3)(x)
x = KL.Dense(8,activation='relu')(x)
x= KL.Dropout(0.2)(x)
x = KL.Dense(8,activation='relu')(x)
x= KL.Dropout(0.1)(x)
x = KL.Dense(8,activation='relu')(x)
x = KL.Dense(3)(x)

model = models.Model([v_input,l_input,m_input,ch0_input,ch1_input],x)
optimizer = optimizers.Adam(lr=0.005)
model.compile(optimizer=optimizer,loss='mse')
model.fit(X,Y,epochs=20,verbose=2,validation_split=0.2)
