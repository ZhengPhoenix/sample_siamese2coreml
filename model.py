from keras.layers import Conv2D, Dense, Flatten,MaxPooling2D, Input, Activation
from keras.models import Model


def Siamese(input_shape=(105, 105, 1)):
    """
    build convnet to use in each siamese 'leg'
    """
    inp = Input(shape=input_shape)
    x = Conv2D(64, (10,10), strides=(2,2), padding='valid')(inp)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (7, 7), strides=(2, 2), padding='valid')(inp)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='valid')(inp)
    x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(8, activation='sigmoid')(x)

    x = Model(inputs=inp, outputs=x)
    return x
