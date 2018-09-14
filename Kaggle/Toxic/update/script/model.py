from keras.layers import Dense, Embedding, Input, SpatialDropout1D
from keras.layers import Bidirectional, Dropout, CuDNNGRU, CuDNNLSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Conv1D
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from script.attlayer import AttentionWeightedAverage


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1), metrics=['accuracy'])
    return model


def get_model_pool(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    # x = Dropout(dropout_rate)(x)
    x, forward_h, backward_h = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True, return_state=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([forward_h, backward_h, avg_pool, max_pool])
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1), metrics=['accuracy'])
    return model


def get_model_deepmoji_style(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size,return_attention=False):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(dropout_rate)(embedding_layer)
    lstm_0_output = Bidirectional(CuDNNLSTM(recurrent_units, return_sequences=True), name="bi_lstm_0")(x)
    lstm_1_output = Bidirectional(CuDNNLSTM(recurrent_units, return_sequences=True), name="bi_lstm_1")(lstm_0_output)
    x = concatenate([lstm_1_output, lstm_0_output, x])
    x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)

    # x = Dropout(dropout_rate)(x)
    # x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1), metrics=['accuracy'])
    return model


def get_model_pool_gru_cnn(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(dropout_rate)(embedding_layer)
    gru_out, forward_h, backward_h = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True, return_state=True))(x)
    conv_out = Conv1D(100, 3, padding='same')(gru_out)

    avg_pool_gru = GlobalAveragePooling1D()(gru_out)
    max_pool_gru = GlobalMaxPooling1D()(gru_out)

    avg_pool_conv = GlobalAveragePooling1D()(conv_out)
    max_pool_conv = GlobalMaxPooling1D()(conv_out)

    x = concatenate([forward_h, backward_h, avg_pool_gru, max_pool_gru, avg_pool_conv, max_pool_conv])

    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model
