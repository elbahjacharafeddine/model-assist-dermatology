# import cv2
import efficientnet.keras as efn
import tensorflow as tf


def get_model():
    ''' This function gets the layers inclunding efficientnet ones. '''

    model_input = tf.keras.Input(shape=(224, 224, 3),
                                 name='img_input')

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    x = efn.EfficientNetB3(include_top=False,
                           weights='noisy-student',
                           input_shape=(224, 224, 3),
                           pooling='avg')(dummy)
    x = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(model_input, x, name='aNetwork')
    model.summary()
    return model
