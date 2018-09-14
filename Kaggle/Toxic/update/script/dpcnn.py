import os
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv1D, BatchNormalization, Dense, Dropout, Embedding, MaxPool1D, Activation, Input, GlobalMaxPooling1D
from keras.layers.merge import add
from keras.optimizers import Adam, SGD, RMSprop

import nltk
import tqdm
import json


def load_glove_embeddings(path):
    with open(path, 'rb') as f:
        lines = [line.split() for line in f]
    words = [l[0] for l in lines]
    vecs = np.array([l[1:] for l in lines])
    word2idx = dict((w, idx) for idx, w in enumerate(words))
    # idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    return word2idx, vecs


def conv_block(inp, n_filters, filter_size, dropout, batch_norm=False, preactivation=True):
    if preactivation:
        x = Activation('relu')(inp)
        x = Conv1D(n_filters, filter_size, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
    else:
        x = Conv1D(n_filters, filter_size, padding='same')(inp)
        x = Activation('relu')(x)
        if batch_norm:
            x = BatchNormalization()(x)
    return Dropout(dropout)(x)


def dp_block(inp, n_filters, filter_size, dropout, batch_norm=False, preactivation=True):
    maxpool = MaxPool1D(pool_size=3, strides=2)(inp)
    x = conv_block(maxpool, n_filters, filter_size, dropout, batch_norm, preactivation)
    x = conv_block(x, n_filters, filter_size, dropout, batch_norm, preactivation)
    return add([maxpool, x])


def dense_block(inp, dense_size, dropout):
    x = Dense(units=dense_size, activation='relu')(inp)
    return Dropout(dropout)(x)


def get_dpcnn_model(embedding_matrix, sequence_length, dropout_rate, dense_size,
                    n_filters=100, filter_size=3, repeat_dp_block=4, repeat_dense_block=1, n_classes=6):

    input_text = Input(shape=(sequence_length,))
    emb = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix], trainable=False)(input_text)
    x = conv_block(emb, n_filters, filter_size, dropout=dropout_rate, preactivation=False)
    x = conv_block(x, n_filters, filter_size, dropout=dropout_rate, preactivation=False)
    emb_resized = Conv1D(n_filters, kernel_size=1, padding='same')(emb)
    x = add([emb_resized, x])

    for _ in range(repeat_dp_block):
        # x = dp_block(x, n_filters, filter_size, dropout=dropout_rate, preactivation=False)
        x = dp_block(x, n_filters, filter_size, dropout=dropout_rate, preactivation=True)

    x = GlobalMaxPooling1D()(x)

    for _ in range(repeat_dense_block):
        x = dense_block(x, dense_size, dropout=dropout_rate)

    preds = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=input_text, outputs=preds)

    model.compile(optimizer=RMSprop(clipvalue=1, clipnorm=1), loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
