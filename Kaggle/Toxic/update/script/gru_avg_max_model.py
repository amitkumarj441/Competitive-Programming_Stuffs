from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop


def get_gru_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x, h = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True, return_state=True))(embedding_layer)
    g_max_pool = GlobalMaxPooling1D()(x)
    g_avg_pool = GlobalAveragePooling1D()(x)
    x = Concatenate([h, g_avg_pool, g_max_pool])
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1), metrics=['accuracy'])
    return model
