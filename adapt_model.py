import numpy as np
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Flatten, Lambda, Input, Embedding
from keras.optimizers import SGD, adam
from keras.layers import ZeroPadding1D, Convolution1D, Convolution2D, GlobalMaxPooling1D
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add, concatenate
from keras.activations import relu
from utils import char_map


def adapt_model(kds_model, input_dim=26, max_query_len=30, embedding_dim=32, max_audio_len=1000):

    input_data = Input(name='the_input', shape=(max_audio_len, input_dim))
    fc1 = kds_model.get_layer('time_distributed_1')(input_data)
    fc2 = kds_model.get_layer('time_distributed_2')(fc1)
    fc3 = kds_model.get_layer('time_distributed_3')(fc2)
    birnn = kds_model.get_layer('bidirectional_1')(fc3)
    char_preds = kds_model.get_layer('out')(birnn)  # Maybe should be get_layer('ypred')?
    pool = Flatten()(char_preds)

    query = Input(shape=(max_query_len,), name='query')
    query_embed = Embedding(input_dim=len(char_map), output_dim=embedding_dim, input_length=max_query_len)(query)
    query_encode = Flatten()(query_embed)

    merged = concatenate([query_encode, pool])
    linear_regression = Dense(64)(merged)
    y_pred = Dense(1, name="y_pred", activation="softmax")(linear_regression)
    model = Model(inputs=[input_data, query], outputs=y_pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
