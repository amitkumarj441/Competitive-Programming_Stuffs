from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import numpy as np
import pandas as pd
from trackml.dataset import load_event
from scipy.spatial.distance import minkowski
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as pp
from sklearn.cluster import DBSCAN
import sklearn.metrics as metrics
from sklearn.metrics import pairwise
from sklearn.cluster.dbscan_ import dbscan
from sklearn.preprocessing import StandardScaler
import hdbscan as _hdbscan


def hdbscan(X, min_samples=1, min_cluster_size=5, cluster_selection_method='eom'):
    cl = _hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method  # eom #leaf
    )
    labels = cl.fit_predict(X)
    labels = labels + 1

    return labels

def do_dbscan_predict(df):

    x  = df.x.values
    y  = df.y.values
    z  = df.z.values
    r  = np.sqrt(x**2+y**2)
    d  = np.sqrt(x**2+y**2+z**2)
    a  = np.arctan2(y,x)
    zr = z/r
    dr = d/r

    results = []
    if 1:

        dz = -0.00012
        for i in range(14):
            print('\rforward %d'%i, end='', flush=True)
            dz = dz + ( i*0.00001)
            aa = a + np.sign(z)*dz*z

            f0 = aa
            f1 = zr
            f2 = aa/zr
            f3 =  1/zr
            f4 = x/d
            f5 = y/d


            X = StandardScaler().fit_transform(np.column_stack([
                f0,f1,f2,f3,f4,f5,
            ]))

            _,l = dbscan(X, eps=0.0035, min_samples=2, algorithm='auto', n_jobs=4)
            #_, l = dbscan(X, eps=0.02, min_samples=1, algorithm='auto', n_jobs=4,metric=minkowski,metric_params={'w':[3,3,3,3,1,1]})
            #l = hdbscan(X, min_samples=4, min_cluster_size=4, cluster_selection_method='eom')

            unique,reverse,count = np.unique(l,return_counts=True,return_inverse=True)
            c = count[reverse]
            c[np.where(l==0)]=0
            c[np.where(c>20)]=0
            results.append((l,c))

            #return l

    #the other direction
    if 0:

        dz = 0.00012
        for i in range(14):
            print('\rbackward %d'%i, end='', flush=True)
            dz = dz - ( i*0.00001)
            aa = a + np.sign(z)*dz*z

            f0 = aa
            f1 = zr
            f2 = aa/zr
            f3 =  1/zr
            f4 = x/d
            f5 = y/d

            X = StandardScaler().fit_transform(np.column_stack([
                f0,f1,f2,f3,f4,f5,
            ]))

            _,l = dbscan(X, eps=0.0035, min_samples=1, algorithm='auto', n_jobs=4)
            #l = hdbscan(X, min_samples=3, min_cluster_size=10, cluster_selection_method='eom')

            unique,reverse,count = np.unique(l,return_counts=True,return_inverse=True)
            c = count[reverse]
            c[np.where(l==0)]=0
            c[np.where(c>20)]=0
            results.append((l,c))


    #----------------------------------------

    labels, counts = results[0]

    for i in range(1,len(results)):
        l,c = results[i]
        idx = np.where((c-counts>0))[0]
        labels[idx] = l[idx] + labels.max()
        counts[idx] = c[idx]

    return labels

def run_dbscan():

    data_dir = '../input/train_sample/train_100_events'

    event_ids = [
        '000001000',  ##

    ]
    sum = 0
    sum_score = 0
    for i, event_id in enumerate(event_ids):
        hits, cells, particles, truth = load_event(data_dir + '/event' + event_id)

        # track_id0 = do_dbscan0_predict(hits)
        # track_id0 = do_hdbscan0_predict(hits)
        track_id1 = do_dbscan_predict(hits)

        # --------------------------------
        track_id = track_id1
    # --------------------------------

    submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
                              data=np.column_stack(([int(event_id), ] * len(hits), hits.hit_id.values, track_id))
                              ).astype(int)

    score = score_event(truth, submission)
    print('[%2d] score : %0.8f' % (i, score))
    sum_score += score
    sum += 1

    print('--------------------------------------')
    print(sum_score / sum)

def run_make_submission():

    data_dir = '../input/test'
    submissions = []

    for event_id, hits, cells in load_dataset(data_dir, parts=['hits', 'cells']):
        print('event_id : ', event_id)

        track_id = do_dbscan_predict(hits)

        # Prepare submission for an event
        submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
            data=np.column_stack(([event_id,]*len(hits), hits.hit_id.values, track_id))
        ).astype(int)
        submissions.append(submission)

    # Create submission file
    submission = pd.concat(submissions, axis=0)
    submission.to_csv('/root/share/project/kaggle/cern/results/submission_dbscan.f00.csv.gz', index=False, compression='gzip')

run_dbscan()
