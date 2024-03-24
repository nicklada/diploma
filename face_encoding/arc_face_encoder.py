from typing import List
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Input,
    Conv2D,
    BatchNormalization,
    PReLU,
    Add,
    Dropout,
    Flatten,
    Dense,
)

from face_detection.retina_face_detector import RetinaFaceDetector
from face_encoding.encoder import Encoder


class ArcFaceEncoder(Encoder):
    def __init__(self):
        self.model = load_model()
        self.face_detector = RetinaFaceDetector()

    def encode(self, img: np.ndarray, is_test=False) -> np.ndarray:
        # обнаружение лица на фото
        detected_face = self.face_detector.detect_and_align(img, is_test=True)
        return self.model(detected_face, training=False).numpy()[0]


def load_model() -> Model:
    base_model = ResNet34()
    inputs = base_model.inputs[0]
    arcface_model = base_model.outputs[0]
    arcface_model = BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
    arcface_model = Dropout(0.4)(arcface_model)
    arcface_model = Flatten()(arcface_model)
    arcface_model = Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(
        arcface_model
    )
    embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(
        arcface_model
    )
    model = Model(inputs, embedding, name=base_model.name)
    model.load_weights("models/arcface_weights.h5")

    return model


def ResNet34() -> Model:
    img_input = Input(shape=(112, 112, 3))

    x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
    x = Conv2D(
        64, 3, strides=1, use_bias=False, kernel_initializer="glorot_normal", name="conv1_conv"
    )(x)
    x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
    x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
    x = stack_fn(x)

    model = Model(img_input, x, name="ResNet34")

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3

    if conv_shortcut:
        shortcut = Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            kernel_initializer="glorot_normal",
            name=name + "_0_conv",
        )(x)
        shortcut = BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn")(x)
    x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
    x = Conv2D(
        filters,
        3,
        strides=1,
        kernel_initializer="glorot_normal",
        use_bias=False,
        name=name + "_1_conv",
    )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn")(x)
    x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)

    x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
    x = Conv2D(
        filters,
        kernel_size,
        strides=stride,
        kernel_initializer="glorot_normal",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn")(x)

    x = Add(name=name + "_add")([shortcut, x])
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def stack_fn(x):
    x = stack1(x, 64, 3, name="conv2")
    x = stack1(x, 128, 4, name="conv3")
    x = stack1(x, 256, 6, name="conv4")
    return stack1(x, 512, 3, name="conv5")
