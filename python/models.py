#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras import backend as K

def build_arm_dnn_model(
        input_shape,
        num_layers = 3, layer_size = 144,
        drop_out_rate = 0.2,
        opt = Adam, initial_learning_rate = 0.0005,
        **kwargs
    ):

    inputs = Input(shape=input_shape)
    shaper = Flatten()(inputs)
    dense_model = Dense(units = layer_size, activation='relu')(shaper)
    dense_model = Dropout(drop_out_rate)(dense_model)
    for i in range(num_layers - 1):
        dense_model = Dense(units = layer_size, activation='relu')(dense_model)
        dense_model = Dropout(drop_out_rate)(dense_model)
    outputs = Dense(31, activation = "softmax")(dense_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_arm_dnn_model_size_s(input_shape):
    return build_arm_dnn_model(input_shape)

def build_arm_dnn_model_size_m(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 3, layer_size = 256)

def build_arm_dnn_model_size_l(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 3, layer_size = 436)

def build_arm_dnn_model_size_xl(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 3, layer_size = 712)

def build_ddnn_model_size_s(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 4, layer_size = 144)

def build_ddnn_model_size_m(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 4, layer_size = 256)

def build_ddnn_model_size_l(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 4, layer_size = 436)

def build_ddnn_model_size_xl(input_shape):
    return build_arm_dnn_model(input_shape, num_layers = 4, layer_size = 712)

def build_arm_cnn_model(
        input_shape,
        layer1_num_filters = 28, layer1_kernel_size = (10, 4), layer1_strides = (1,1),
        layer2_num_filters = 30, layer2_kernel_size = (10, 4), layer2_strides = (2,1),
        linear_layer_num_units = 16, fully_connected_layer_num_units = 128,
        drop_out_rate = 0.2,
        opt = Adam, initial_learning_rate = 0.0005,
        **kwargs
    ):
    inputs = Input(shape = input_shape)
    shaper = Reshape((input_shape[0], input_shape[1], 1))(inputs)

    cnn_model = Conv2D(filters = layer1_num_filters, kernel_size = layer1_kernel_size, strides = layer1_strides, padding='valid')(shaper)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(drop_out_rate)(cnn_model)

    cnn_model = Conv2D(filters = layer2_num_filters, kernel_size = layer2_kernel_size, strides = layer2_strides, padding='valid')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(drop_out_rate)(cnn_model)

    cnn_model = Flatten()(cnn_model)

    dense_model = Dense(units = linear_layer_num_units, activation = 'linear')(cnn_model)
    dense_model = Dense(units = fully_connected_layer_num_units)(dense_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Activation('relu')(dense_model)
    dense_model = Dropout(drop_out_rate)(dense_model)

    outputs = Dense(units = 31, activation = "softmax")(dense_model)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_arm_cnn_model_size_s(input_shape):
    return build_arm_cnn_model(
        input_shape,
        layer1_num_filters = 28, layer1_kernel_size = (10, 4), layer1_strides = (1,1),
        layer2_num_filters = 30, layer2_kernel_size = (10, 4), layer2_strides = (2,1),
        linear_layer_num_units = 16, fully_connected_layer_num_units = 128
        )

def build_arm_cnn_model_size_m(input_shape):
    return build_arm_cnn_model(
        input_shape,
        layer1_num_filters = 64, layer1_kernel_size = (10, 4), layer1_strides = (1,1),
        layer2_num_filters = 48, layer2_kernel_size = (10, 4), layer2_strides = (2,1),
        linear_layer_num_units = 16, fully_connected_layer_num_units = 128
        )

def build_arm_cnn_model_size_l(input_shape):
    return build_arm_cnn_model(
        input_shape,
        layer1_num_filters = 60, layer1_kernel_size = (10, 4), layer1_strides = (1,1),
        layer2_num_filters = 76, layer2_kernel_size = (10, 4), layer2_strides = (2,1),
        linear_layer_num_units = 58, fully_connected_layer_num_units = 128
        )

def build_arm_cnn_model_size_xl(input_shape):
    return build_arm_cnn_model(
        input_shape,
        layer1_num_filters = 128, layer1_kernel_size = (10, 4), layer1_strides = (1,1),
        layer2_num_filters = 128, layer2_kernel_size = (10, 4), layer2_strides = (2,1),
        linear_layer_num_units = 64, fully_connected_layer_num_units = 128
        )

def build_arm_rnn_model(
        input_shape,
        rnn_cell = LSTM, rnn_cell_num_units = 118,
        drop_out_rate = 0.2,
        opt = Adam, initial_learning_rate = 0.0005,
        **kwargs
    ):
    inputs = Input(shape=input_shape)

    rnn_model = rnn_cell(units = rnn_cell_num_units)(inputs)
    rnn_model = Dropout(drop_out_rate)(rnn_model)

    outputs = Dense(31, activation = "softmax")(rnn_model)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_arm_lstm_model_size_s(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = LSTM, rnn_cell_num_units = 118)

def build_arm_lstm_model_size_m(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = LSTM, rnn_cell_num_units = 214)

def build_arm_lstm_model_size_l(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = LSTM, rnn_cell_num_units = 344)

def build_arm_lstm_model_size_xl(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = LSTM, rnn_cell_num_units = 512)

def build_arm_gru_model_size_s(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = GRU, rnn_cell_num_units = 154)

def build_arm_gru_model_size_m(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = GRU, rnn_cell_num_units = 250)

def build_arm_gru_model_size_l(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = GRU, rnn_cell_num_units = 400)

def build_arm_gru_model_size_xl(input_shape):
    return build_arm_rnn_model(input_shape, rnn_cell = GRU, rnn_cell_num_units = 616)

def build_arm_crnn_model(
        input_shape,
        cnn_layer_num_filters = 48, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2),
        num_rnn_layers = 2, rnn_cell = GRU,
        rnn_layer1_num_cells = 60, rnn_layer2_num_cells = 60,
        rnn_layer3_num_cells = 60, rnn_layer4_num_cells = 60,
        fully_connected_layer_num_units = 84,
        drop_out_rate = 0.2,
        opt = Adam, initial_learning_rate = 0.0005,
        **kwargs
    ):
    inputs = Input(shape=input_shape)
    shaper = Reshape((input_shape[0], input_shape[1], 1))(inputs)

    cnn_model = Conv2D(filters = cnn_layer_num_filters, kernel_size = cnn_layer_kernel_size, strides = cnn_layer_strides, padding='valid')(shaper)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(drop_out_rate)(cnn_model)

    cnn_model_width, cnn_model_height, cnn_model_depth = [int(val) for val in cnn_model.shape[1:]]
    permuter = Permute((1, 3, 2))(cnn_model)
    reshaper = Reshape((cnn_model_width, cnn_model_height * cnn_model_depth))(permuter)

    rnn_model = rnn_cell(units = rnn_layer1_num_cells, return_sequences = True)(reshaper)
    if num_rnn_layers == 2:
        rnn_model = rnn_cell(units = rnn_layer2_num_cells, return_sequences = False)(rnn_model)

    if num_rnn_layers == 3:
        rnn_model = rnn_cell(units = rnn_layer2_num_cells, return_sequences = True)(rnn_model)
        rnn_model = rnn_cell(units = rnn_layer3_num_cells, return_sequences = False)(rnn_model)

    if num_rnn_layers == 4:
        rnn_model = rnn_cell(units = rnn_layer2_num_cells, return_sequences = True)(rnn_model)
        rnn_model = rnn_cell(units = rnn_layer3_num_cells, return_sequences = True)(rnn_model)
        rnn_model = rnn_cell(units = rnn_layer4_num_cells, return_sequences = False)(rnn_model)

    dense_model = Dense(units = fully_connected_layer_num_units)(rnn_model)
    dense_model = Activation('relu')(dense_model)
    dense_model = Dropout(drop_out_rate)(dense_model)

    outputs = Dense(units = 31, activation = "softmax")(dense_model)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_arm_crnn_model_size_s(input_shape):
    return build_arm_crnn_model(
        input_shape,
        cnn_layer_num_filters = 48, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2),
        rnn_cell = GRU, rnn_layer1_num_cells = 60, rnn_layer2_num_cells = 60,
        fully_connected_layer_num_units = 84
        )

def build_arm_crnn_model_size_m(input_shape):
    return build_arm_crnn_model(
        input_shape,
        cnn_layer_num_filters = 128, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2),
        rnn_cell = GRU, rnn_layer1_num_cells = 76, rnn_layer2_num_cells = 76,
        fully_connected_layer_num_units = 164
        )

def build_arm_crnn_model_size_l(input_shape):
    return build_arm_crnn_model(
        input_shape,
        cnn_layer_num_filters = 100, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,1),
        rnn_cell = GRU, rnn_layer1_num_cells = 136, rnn_layer2_num_cells = 136,
        fully_connected_layer_num_units = 188
        )

def build_arm_crnn_model_size_xl(input_shape):
    return build_arm_crnn_model(
        input_shape,
        cnn_layer_num_filters = 168, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,1),
        rnn_cell = GRU, rnn_layer1_num_cells = 168, rnn_layer2_num_cells = 168,
        fully_connected_layer_num_units = 256
        )

def build_my_crnn_model_size_1(input_shape):
    return build_arm_crnn_model(
        input_shape,
        cnn_layer_num_filters = 32, cnn_layer_kernel_size = (4, 16), cnn_layer_strides = (4, 4),
        rnn_cell = GRU, rnn_layer1_num_cells = 128, rnn_layer2_num_cells = 128,
        fully_connected_layer_num_units = 256
        )

def build_arm_dscnn_model(
        input_shape,
        layer1_num_filters = 64, layer1_kernel_size = (10, 4), layer1_strides = (2,2),
        num_dsc_layer = 3, dsc_num_filters = 64, dsc_kernel_size = 3, dsc_strides = 1,
        drop_out_rate = 0.2,
        opt = Adam, initial_learning_rate = 0.0005,
        **kwargs
    ):

    inputs = Input(shape=input_shape)
    shaper = Reshape((input_shape[0], input_shape[1], 1))(inputs)

    cnn_model = Conv2D(filters = layer1_num_filters, kernel_size = layer1_kernel_size, strides = layer1_strides, padding = 'same')(shaper)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(drop_out_rate)(cnn_model)

    for i in range(num_dsc_layer):
        cnn_model = SeparableConv2D(filters = dsc_num_filters, kernel_size = dsc_kernel_size, strides = dsc_strides, padding = 'same')(cnn_model)
        cnn_model = BatchNormalization()(cnn_model)
        cnn_model = Activation('relu')(cnn_model)
        cnn_model = Dropout(drop_out_rate)(cnn_model)

    cnn_model = GlobalAveragePooling2D()(cnn_model)
    outputs = Dense(units = 31, activation = "softmax")(cnn_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_arm_dscnn_model_size_s(input_shape):
    return build_arm_dscnn_model(
        input_shape,
        layer1_num_filters = 64, layer1_kernel_size = (10, 4), layer1_strides = (2,2),
        num_dsc_layer = 4, dsc_num_filters = 64, dsc_kernel_size = 3, dsc_strides = 1,
        )

def build_arm_dscnn_model_size_m(input_shape):
    return build_arm_dscnn_model(
        input_shape,
        layer1_num_filters = 172, layer1_kernel_size = (10, 4), layer1_strides = (2,1),
        num_dsc_layer = 4, dsc_num_filters = 172, dsc_kernel_size = 3, dsc_strides = 1,
        )

def build_arm_dscnn_model_size_l(input_shape):
    return build_arm_dscnn_model(
        input_shape,
        layer1_num_filters = 276, layer1_kernel_size = (10, 4), layer1_strides = (2,1),
        num_dsc_layer = 4, dsc_num_filters = 276, dsc_kernel_size = 3, dsc_strides = 1,
        )

def build_arm_dscnn_model_size_xl(input_shape):
    return build_arm_dscnn_model(
        input_shape,
        layer1_num_filters = 384, layer1_kernel_size = (10, 4), layer1_strides = (2,1),
        num_dsc_layer = 4, dsc_num_filters = 384, dsc_kernel_size = 3, dsc_strides = 1,
        )

def build_arm_dscnn_model_size_t(input_shape):
    return build_arm_dscnn_model(
        input_shape,
        layer1_num_filters = 192, layer1_kernel_size = (3, 12), layer1_strides = (2,2),
        num_dsc_layer = 4, dsc_num_filters = 192, dsc_kernel_size = 3, dsc_strides = 1,
        opt = Adam, initial_learning_rate = 2,
        drop_out_rate = 0.2
        )

def build_gg4_model(
        input_shape,
        cnn_block_1_num_filters = 16, cnn_block_1_kernel_size = (2,8), cnn_block_1_strides = (1,1), cnn_block_1_pool_size = 2,
        cnn_block_2_num_filters = 32, cnn_block_2_kernel_size = (2,8), cnn_block_2_strides = (1,1), cnn_block_2_pool_size = 2,
        cnn_block_3_num_filters = 64, cnn_block_3_kernel_size = (2,8), cnn_block_3_strides = (1,1), cnn_block_3_pool_size = 2,
        cnn_block_4_num_filters = 128, cnn_block_4_kernel_size = (2,8), cnn_block_4_strides = (1,1), cnn_block_4_pool_size = 2,
        num_fully_connected_layers = 2, fully_connected_layer_num_units = 256,
        drop_out_rate = 0.5,
        opt = Adadelta, initial_learning_rate = 1,
        **kwargs
    ):

    inputs = Input(shape=input_shape)
    shaper = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    cnn_model = Conv2D(filters = cnn_block_1_num_filters, kernel_size = cnn_block_1_kernel_size, strides = cnn_block_1_strides, padding='same')(shaper)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv2D(filters = cnn_block_1_num_filters, kernel_size = cnn_block_1_kernel_size, strides = cnn_block_1_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling2D(pool_size = cnn_block_1_pool_size)(cnn_model)

    cnn_model = Conv2D(filters = cnn_block_2_num_filters, kernel_size = cnn_block_2_kernel_size, strides = cnn_block_2_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv2D(filters = cnn_block_2_num_filters, kernel_size = cnn_block_2_kernel_size, strides = cnn_block_2_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling2D(pool_size = cnn_block_2_pool_size)(cnn_model)

    cnn_model = Conv2D(filters = cnn_block_3_num_filters, kernel_size = cnn_block_3_kernel_size, strides = cnn_block_3_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv2D(filters = cnn_block_3_num_filters, kernel_size = cnn_block_3_kernel_size, strides = cnn_block_3_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling2D(pool_size = cnn_block_3_pool_size)(cnn_model)

    cnn_model = Conv2D(filters = cnn_block_4_num_filters, kernel_size = cnn_block_4_kernel_size, strides = cnn_block_4_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv2D(filters = cnn_block_4_num_filters, kernel_size = cnn_block_4_kernel_size, strides = cnn_block_4_strides, padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling2D(pool_size = cnn_block_4_pool_size)(cnn_model)

    global_max_pool = GlobalMaxPooling2D()(cnn_model)
    global_avg_pool = GlobalAveragePooling2D()(cnn_model)

    cnn_model = concatenate([global_max_pool, global_avg_pool])

    if num_fully_connected_layers > 0:
        dense_model = Dense(units = fully_connected_layer_num_units, activation = 'relu')(cnn_model)
        dense_model = BatchNormalization()(dense_model)
        dense_model = Dropout(drop_out_rate)(dense_model)
        for i in range(1, num_fully_connected_layers):
            dense_model = Dense(units = fully_connected_layer_num_units, activation = 'relu')(dense_model)
            dense_model = BatchNormalization()(dense_model)
            dense_model = Dropout(drop_out_rate)(dense_model)

        outputs = Dense(31, activation = "softmax")(dense_model)
    else:
        outputs = Dense(31, activation = "softmax")(cnn_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_gg4_model_size_1(input_shape):
    # for logspectrogram_25_18.75
    return build_gg4_model(
        input_shape,
        cnn_block_1_num_filters = 16, cnn_block_1_kernel_size = (2,8), cnn_block_1_strides = (1,1), cnn_block_1_pool_size = 2,
        cnn_block_2_num_filters = 32, cnn_block_2_kernel_size = (2,8), cnn_block_2_strides = (1,1), cnn_block_2_pool_size = 2,
        cnn_block_3_num_filters = 64, cnn_block_3_kernel_size = (2,8), cnn_block_3_strides = (1,1), cnn_block_3_pool_size = 2,
        cnn_block_4_num_filters = 128, cnn_block_4_kernel_size = (2,8), cnn_block_4_strides = (1,1), cnn_block_4_pool_size = 2,
        num_fully_connected_layers = 2, fully_connected_layer_num_units = 256,
        drop_out_rate = 0.5
        )

def build_gg4_model_size_2(input_shape):
    return build_gg4_model(
        input_shape,
        cnn_block_1_num_filters = 16, cnn_block_1_kernel_size = (2,5), cnn_block_1_strides = (1,1), cnn_block_1_pool_size = 2,
        cnn_block_2_num_filters = 32, cnn_block_2_kernel_size = (2,5), cnn_block_2_strides = (1,1), cnn_block_2_pool_size = 2,
        cnn_block_3_num_filters = 64, cnn_block_3_kernel_size = (2,5), cnn_block_3_strides = (1,1), cnn_block_3_pool_size = 2,
        cnn_block_4_num_filters = 128, cnn_block_4_kernel_size = (2,5), cnn_block_4_strides = (1,1), cnn_block_4_pool_size = 2,
        num_fully_connected_layers = 2, fully_connected_layer_num_units = 256,
        drop_out_rate = 0.5
        )

def build_gg4_model_size_s(input_shape):
    return build_gg4_model(
        input_shape,
        cnn_block_1_num_filters = 16, cnn_block_1_kernel_size = (5, 2), cnn_block_1_strides = (1,1), cnn_block_1_pool_size = 2,
        cnn_block_2_num_filters = 32, cnn_block_2_kernel_size = (5, 2), cnn_block_2_strides = (1,1), cnn_block_2_pool_size = 2,
        cnn_block_3_num_filters = 64, cnn_block_3_kernel_size = (5, 2), cnn_block_3_strides = (1,1), cnn_block_3_pool_size = 2,
        cnn_block_4_num_filters = 128, cnn_block_4_kernel_size = (5, 2), cnn_block_4_strides = (1,1), cnn_block_4_pool_size = 2,
        num_fully_connected_layers = 2, fully_connected_layer_num_units = 256,
        drop_out_rate = 0.5
        )

def build_gg4_model_size_3(input_shape):
    return build_gg4_model(
        input_shape,
        cnn_block_1_num_filters = 18, cnn_block_1_kernel_size = (5,2), cnn_block_1_strides = (1,1), cnn_block_1_pool_size = 2,
        cnn_block_2_num_filters = 36, cnn_block_2_kernel_size = (5,2), cnn_block_2_strides = (1,1), cnn_block_2_pool_size = 2,
        cnn_block_3_num_filters = 72, cnn_block_3_kernel_size = (5,2), cnn_block_3_strides = (1,1), cnn_block_3_pool_size = 2,
        cnn_block_4_num_filters = 144, cnn_block_4_kernel_size = (5,2), cnn_block_4_strides = (1,1), cnn_block_4_pool_size = 2,
        num_fully_connected_layers = 2, fully_connected_layer_num_units = 128,
        drop_out_rate = 0.5
        )

def build_join_model_size_t(
        input_shape,
        opt = Adadelta,
        initial_learning_rate = 0.001
    ):

    inputs = Input(shape = input_shape)

    cnn_inputs = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    cnn_model = Conv2D(filters = 32, kernel_size = (2, 8), strides = (1, 1), padding='same')(cnn_inputs)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = Conv2D(filters = 32, kernel_size = (2, 8), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = MaxPooling2D(pool_size = (2, 2))(cnn_model)

    cnn_model = Conv2D(filters = 64, kernel_size = (2, 8), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = Conv2D(filters = 64, kernel_size = (2, 8), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = MaxPooling2D(pool_size = (2, 2))(cnn_model)

    cnn_model = Conv2D(filters = 128, kernel_size = (2, 8), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = Conv2D(filters = 128, kernel_size = (2, 8), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = MaxPooling2D(pool_size = (2, 2))(cnn_model)

    global_max_pool = GlobalMaxPooling2D()(cnn_model)
    global_avg_pool = GlobalAveragePooling2D()(cnn_model)

    rnn_model = GRU(units = 256)(inputs)
    rnn_model = Dropout(0.3)(rnn_model)

    concat = concatenate([global_max_pool, global_avg_pool, rnn_model])

    dense_model = Dense(units = 512, activation = 'relu')(concat)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.3)(dense_model)
    dense_model = Dense(units = 512, activation = 'relu')(dense_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.3)(dense_model)

    outputs = Dense(31, activation = "softmax")(dense_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_join_model_size_s(
        input_shape,
        opt = Adadelta,
        initial_learning_rate = 0.1
    ):
    inputs = Input(shape = input_shape)

    cnn_inputs = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    cnn_model = Conv2D(filters = 32, kernel_size = (2, 5), strides = (1, 1), padding='same')(cnn_inputs)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = Conv2D(filters = 32, kernel_size = (2, 5), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = MaxPooling2D(pool_size = (2, 2))(cnn_model)

    cnn_model = Conv2D(filters = 64, kernel_size = (2, 5), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = Conv2D(filters = 64, kernel_size = (2, 5), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = MaxPooling2D(pool_size = (2, 2))(cnn_model)

    cnn_model = Conv2D(filters = 128, kernel_size = (2, 5), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = Conv2D(filters = 128, kernel_size = (2, 5), strides = (1, 1), padding='same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    cnn_model = MaxPooling2D(pool_size = (2, 2))(cnn_model)

    global_max_pool = GlobalMaxPooling2D()(cnn_model)
    global_avg_pool = GlobalAveragePooling2D()(cnn_model)

    rnn_model = GRU(units = 256)(inputs)
    rnn_model = Dropout(0.3)(rnn_model)

    concat = concatenate([global_max_pool, global_avg_pool, rnn_model])
    dense_model = Dense(units = 512, activation = 'relu')(concat)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.3)(dense_model)
    dense_model = Dense(units = 512, activation = 'relu')(dense_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.3)(dense_model)

    outputs = Dense(31, activation = "softmax")(dense_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_1dcnn_model(
        input_shape = (16000,1),
        num_cnn_blocks = 9,
        cnn_layer_kernel_sizes = [2 ** (i + 2) for i in range(9)],
        kernel_size = [4 for i in range(9)],
        strides = [1 for i in range(9)],
        pooling_sizes = [2 for i in range(9)],
        global_pooling_method = 'max',
        num_fully_connected_layers = 2,
        fully_connected_layer_num_units = 1024,
        drop_out_rate = 0.2,
        opt = Adadelta, initial_learning_rate = 1,
        **kwargs
    ):
    inputs = Input(shape = input_shape)
    for i in range(num_cnn_blocks):
        if i == 0:
            cnn_model = Conv1D(filters = cnn_layer_kernel_sizes[i], kernel_size = kernel_size[i], strides = strides[i], padding = 'same')(inputs)
        else:
            cnn_model = Conv1D(filters = cnn_layer_kernel_sizes[i], kernel_size = kernel_size[i], strides = strides[i], padding = 'same')(cnn_model)
        cnn_model = BatchNormalization()(cnn_model)
        cnn_model = Activation('relu')(cnn_model)
        cnn_model = Conv1D(filters = cnn_layer_kernel_sizes[i], kernel_size = kernel_size[i], strides = strides[i], padding = 'same')(cnn_model)
        cnn_model = BatchNormalization()(cnn_model)
        cnn_model = Activation('relu')(cnn_model)
        cnn_model = MaxPooling1D(pooling_sizes[i])(cnn_model)

    if global_pooling_method == 'max':
        cnn_model = GlobalMaxPooling1D()(cnn_model)
    if global_pooling_method == 'avg':
        cnn_model = GlobalAveragePooling1D()(cnn_model)
    if global_pooling_method == 'both':
        global_max_pool = GlobalMaxPooling1D()(cnn_model)
        global_avg_pool = GlobalAveragePooling1D()(cnn_model)
        cnn_model = concatenate([global_max_pool, global_avg_pool])

    dense_model = Dense(fully_connected_layer_num_units, activation = 'relu')(cnn_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(drop_out_rate)(dense_model)

    for j in range(1, num_fully_connected_layers):
        dense_model = Dense(fully_connected_layer_num_units, activation = 'relu')(dense_model)
        dense_model = BatchNormalization()(dense_model)
        dense_model = Dropout(drop_out_rate)(dense_model)

    outputs = Dense(31, activation='softmax')(dense_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["categorical_accuracy"])
    return model

def build_1dcnn_model_size_1(input_shape = (16000,1)):
    return build_1dcnn_model(
            input_shape = (16000,1),
            num_cnn_blocks = 9,
            cnn_layer_kernel_sizes = [2 ** (i + 2) for i in range(9)],
            kernel_size = [4 for i in range(9)],
            strides = [1 for i in range(9)],
            pooling_sizes = [2 for i in range(9)],
            global_pooling_method = 'both',
            num_fully_connected_layers = 2,
            fully_connected_layer_num_units = 1024,
            drop_out_rate = 0.2
        )

def build_1dcnn_model_size_l(input_shape = (16000, 1)):
    inputs = Input(shape = input_shape)

    # cnn block 1
    cnn_model = Conv1D(filters = 8, kernel_size = 3, strides = 1, padding = 'same')(inputs)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 8, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)

    # cnn block 2
    cnn_model = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 16, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)

    # cnn block 3
    cnn_model = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 4
    cnn_model = Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 5
    cnn_model = Conv1D(filters = 128, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 128, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 6
    cnn_model = Conv1D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 7
    cnn_model = Conv1D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 8
    cnn_model = Conv1D(filters = 1024, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 1024, kernel_size = 3, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    global_avg_pool = GlobalAveragePooling1D()(cnn_model)

    dense_model = Dense(512, activation = 'relu')(global_avg_pool)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.2)(dense_model)

    dense_model = Dense(256, activation = 'relu')(dense_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.2)(dense_model)

    outputs = Dense(31, activation='softmax')(dense_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adadelta(lr = 1)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["categorical_accuracy"])
    return model

def build_1dcnn_model_size_xl(input_shape = (16000, 1)):
    inputs = Input(shape = input_shape)

    # cnn block 1
    cnn_model = Conv1D(filters = 4, kernel_size = 8, strides = 1, padding = 'same')(inputs)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 4, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)

    # cnn block 2
    cnn_model = Conv1D(filters = 8, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 8, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)

    # cnn block 3
    cnn_model = Conv1D(filters = 16, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 16, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 4
    cnn_model = Conv1D(filters = 32, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 32, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 5
    cnn_model = Conv1D(filters = 64, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 64, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 6
    cnn_model = Conv1D(filters = 128, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 128, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 128, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 7
    cnn_model = Conv1D(filters = 256, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 256, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 256, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 7
    cnn_model = Conv1D(filters = 512, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 512, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 512, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    # cnn block 8
    cnn_model = Conv1D(filters = 1024, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)
    cnn_model = Conv1D(filters = 1024, kernel_size = 8, strides = 1, padding = 'same')(cnn_model)
    cnn_model = BatchNormalization()(cnn_model)
    cnn_model = Activation('relu')(cnn_model)

    cnn_model = MaxPooling1D(pool_size = 2, strides = 2)(cnn_model)
    cnn_model = Dropout(0.1)(cnn_model)

    global_max_pool = GlobalMaxPooling1D()(cnn_model)
    global_avg_pool = GlobalAveragePooling1D()(cnn_model)
    cnn_model = concatenate([global_max_pool, global_avg_pool])

    dense_model = Dense(1024, activation = 'relu')(cnn_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.2)(dense_model)

    dense_model = Dense(512, activation = 'relu')(dense_model)
    dense_model = BatchNormalization()(dense_model)
    dense_model = Dropout(0.2)(dense_model)

    outputs = Dense(31, activation='softmax')(dense_model)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adadelta(lr = 1)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["categorical_accuracy"])
    return model

def build_resnet_model(
        input_shape,
        bottle_neck = False,
        cnn_layer_num_filters = 64, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2), cnn_layer_pool_size = 2,
        repetitions = [2,2,2,2], res_block_initial_num_filters = 4, res_conv_block_kernel_size = (4, 2),
        drop_out_rate = 0.2,
        opt = Adam, initial_learning_rate = 0.5,
        **kwargs
    ):

    inputs = Input(shape = input_shape)
    reshape = Reshape((input_shape[0], input_shape[1], 1))(inputs)

    conv1 = Conv2D(filters = cnn_layer_num_filters, kernel_size = cnn_layer_kernel_size, strides = cnn_layer_strides, kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(reshape)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Dropout(drop_out_rate)(conv1)
    conv1 = MaxPooling2D(pool_size = cnn_layer_pool_size)(conv1)

    res_block = conv1
    filters = res_block_initial_num_filters

    for layer_num, repetition_num in enumerate(repetitions):

        for reptition in range(repetition_num):
            res_strides = (1,1)
            if layer_num > 0  and reptition == 0:
                res_strides = (2,2)

            if layer_num == 0 and reptition == 0:
                if bottle_neck:
                    conv = Conv2D(filters = filters, kernel_size = (1,1), strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(res_block)
                else:
                    conv = Conv2D(filters = filters, kernel_size = res_conv_block_kernel_size, strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(res_block)
            else:
                conv = BatchNormalization()(res_block)
                conv = Activation("relu")(conv)
                conv = Dropout(drop_out_rate)(conv)
                if bottle_neck:
                    conv = Conv2D(filters = filters, kernel_size = (1,1), strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(conv)
                    conv = BatchNormalization()(conv)
                    conv = Activation("relu")(conv)
                    conv = Dropout(drop_out_rate)(conv)
                    conv = Conv2D(filters = filters, kernel_size = res_conv_block_kernel_size, strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))
                else:
                    conv = Conv2D(filters = filters, kernel_size = res_conv_block_kernel_size, strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(conv)

            residual = BatchNormalization()(conv)
            residual = Activation("relu")(residual)
            residual = Dropout(drop_out_rate)(residual)
            if bottle_neck:
                residual = Conv2D(filters = filters * 4, kernel_size = (1,1), strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(residual)
            else:
                residual = Conv2D(filters = filters, kernel_size = res_conv_block_kernel_size, strides = res_strides, padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0001))(residual)

            res_block_shape = K.int_shape(res_block)
            residual_shape = K.int_shape(residual)
            stride_width = int(round(res_block_shape[1] / residual_shape[1]))
            stride_height = int(round(res_block_shape[2] / residual_shape[2]))
            equal_channels = res_block_shape[3] == residual_shape[3]

            short_cut = res_block
            if stride_width > 1 or stride_height > 1 or not equal_channels:
                short_cut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_width, stride_height), padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(0.0001))(res_block)
            res_block = concatenate([short_cut, residual])
        filters *= 2

    res_block = BatchNormalization()(res_block)
    res_block = Activation("relu")(res_block)

    res_block_shape = K.int_shape(res_block)
    avg_pool = AveragePooling2D(pool_size=(res_block_shape[1], res_block_shape[2]), strides=(1, 1))(res_block)

    flatten = Flatten()(avg_pool)
    outputs = Dense(units=31, kernel_initializer="he_normal", activation="softmax")(flatten)
    model = Model(inputs = inputs,outputs = outputs)
    optimizer = opt(lr = initial_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["categorical_accuracy"])
    return model

def build_resnet_model_size_18(input_shape):
    return build_resnet_model(
        input_shape,
        cnn_layer_num_filters = 64, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2), cnn_layer_pool_size = 2,
        repetitions = [2,2,2,2],
        res_block_initial_num_filters = 64,
        res_conv_block_kernel_size = (4, 2),
        opt = Adadelta, initial_learning_rate = 0.5
    )

def build_resnet_model_size_18t(input_shape):
    return build_resnet_model(
        input_shape,
        cnn_layer_num_filters = 32, cnn_layer_kernel_size = (3, 12), cnn_layer_strides = (2,2), cnn_layer_pool_size = 1,
        repetitions = [2,2,2,2],
        res_block_initial_num_filters = 32,
        res_conv_block_kernel_size = (2, 5),
        opt = Adadelta, initial_learning_rate = 1
    )

def build_resnet_model_size_34(input_shape):
    return build_resnet_model(
        input_shape,
        cnn_layer_num_filters = 64, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2), cnn_layer_pool_size = 2,
        repetitions = [3,4,6,3],
        res_block_initial_num_filters = 64,
        res_conv_block_kernel_size = (4, 2),
        opt = Adadelta, initial_learning_rate = 0.5
    )

def build_resnet_model_size_34t(input_shape):
    return build_resnet_model(
        input_shape,
        cnn_layer_num_filters = 32, cnn_layer_kernel_size = (3, 12), cnn_layer_strides = (2,2), cnn_layer_pool_size = 1,
        repetitions = [3,4,6,3],
        res_block_initial_num_filters = 32,
        res_conv_block_kernel_size = (2, 5),
        opt = Adadelta, initial_learning_rate = 1
    )

def build_resnet_model_size_50(input_shape):
    return build_resnet_model(
        input_shape,
        bottle_neck = True,
        cnn_layer_num_filters = 64, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2), cnn_layer_pool_size = 2,
        repetitions = [3,4,6,3],
        res_block_initial_num_filters = 64,
        res_conv_block_kernel_size = (4, 2),
        opt = Adadelta, initial_learning_rate = 0.5
    )

def build_resnet_model_size_101(input_shape):
    return build_resnet_model(
        input_shape,
        bottle_neck = True,
        cnn_layer_num_filters = 64, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2), cnn_layer_pool_size = 2,
        repetitions = [3,4,23,3],
        res_block_initial_num_filters = 64,
        res_conv_block_kernel_size = (4, 2),
        opt = Adadelta, initial_learning_rate = 0.5
    )


def build_resnet_model_size_152(input_shape):
    return build_resnet_model(
        input_shape,
        bottle_neck = True,
        cnn_layer_num_filters = 64, cnn_layer_kernel_size = (10, 4), cnn_layer_strides = (2,2), cnn_layer_pool_size = 2,
        repetitions = [3,8,36,3],
        res_block_initial_num_filters = 64,
        res_conv_block_kernel_size = (4, 2),
        opt = Adadelta, initial_learning_rate = 0.5
    )
