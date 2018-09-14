from keras.models import Model
from keras.layers import CuDNNGRU, Input, Embedding, Dense, Dropout, Bidirectional, SpatialDropout1D
from keras.optimizers import Adam, RMSprop
from script.attention import Attention


def get_model_att_lstm(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    inp = Input(shape=(sequence_length,))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Attention()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inp, x)
    model.compile(RMSprop(lr=0.001, clipvalue=1, clipnorm=1), loss='binary_crossentropy', metrics=['accuracy'])
    return model
