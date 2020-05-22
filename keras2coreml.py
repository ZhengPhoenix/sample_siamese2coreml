from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras import backend as K
from model import Siamese
import os

model_name = 'weight.h5'
weight_path = os.path.join('model', model_name)

# define model structure
input_shape = (105, 105, 1)
left_input = Input(shape=input_shape)
right_input = Input(shape=input_shape)

siamese_net = Siamese(input_shape=input_shape)

encoded_l = siamese_net(left_input)
encoded_r = siamese_net(right_input)

L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1, activation='sigmoid')(L1_distance)

net = Model(inputs=[left_input, right_input], outputs=prediction)
net.summary()
net.save(weight_path)

# convert to coreml
import coremltools
coreml_model = coremltools.converters.tensorflow.convert(filename=weight_path,
                                                         input_names=['input_1', 'input_2'],
                                                         output_names='dense_2',
                                                         image_input_names=['input_1', 'input_2'],
                                                         input_name_shape_dict={'input_1': [105, 105, 1],
                                                                                'input_2': [105, 105, 1]})
coreml_model.save(str(weight_path[:-3] + '.mlmodel'))
