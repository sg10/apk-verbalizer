from keras import Model
from keras.layers import Input, Dropout, Dense

model_config_class = None


def model_for_tf_idfs(num_inputs, num_outputs):
    inp = Input(shape=(num_inputs,))
    layer = inp

    for n in model_config_class.layers:
        layer = Dense(n, activation='relu')(layer)
        layer = Dropout(model_config_class.dropout)(layer)

    out = Dense(num_outputs, activation='linear')(layer)

    return Model(inputs=inp, outputs=out)
