import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class RussianHouse:

    def __init__(self):
        pass

    def addFeatures(self, df):
    
        pass

    def reduceDimension(self, train, test):
        imp = Imputer(strategy="mean", axis=0)
        train = imp.fit_transform(train)
        test= imp.fit_transform(test)

        pca = PCA(n_components=50)
        pca.fit(train)
        train = pca.transform(train)

        pca = PCA(n_components=50)
        pca.fit(test)
        test = pca.transform(test)


    def addComplexFeatures(self, train, test, featureName):
        lb = preprocessing.LabelEncoder()
        lb.fit(list(train[featureName].values))
        train[featureName] = lb.transform(list(train[featureName].values))

        lb = preprocessing.LabelEncoder()
        lb.fit(list(test[featureName].values))
        test[featureName] = lb.transform(list(test[featureName].values))

    def transform(self, train, test):

        self.addFeatures(train)
        self.addFeatures(test)

        train.drop(["id", "timestamp"], axis=1, inplace=True)
        test.drop(["id", "timestamp"], axis=1, inplace=True)

        for x in train.columns:
            if train[x].dtype == 'object':
                self.addComplexFeatures(train,test,x)

        for x in train.columns:
            if train[x].dtypes == "object":
                train.drop(x, axis=1, inplace=True)

        for x in test.columns:
            if test[x].dtypes == "object":
                test.drop(x, axis=1, inplace=True)

        return train,test

    def corr_plot(self, dataframe, top_n, target, fig_x, fig_y):
        train = dataframe.copy()

        corrmat = dataframe.corr()
        # top_n - top n correlations +1 since price is included
        top_n = top_n + 1
        cols = corrmat.nlargest(top_n, target)[target].index
        cm = np.corrcoef(train[cols].values.T)
        f, ax = plt.subplots(figsize=(fig_x, fig_y))
        sns.set(font_scale=1.25)
        cmap = plt.cm.viridis
        hm = sns.heatmap(cm, cbar=False, annot=True, square=True, cmap=cmap, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return cols
