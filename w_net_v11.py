from __future__ import print_function

from keras import backend as K
from keras.layers import Dropout
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Lambda
from keras.layers.convolutional import SeparableConv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.engine.topology import Layer


K.set_image_data_format('channels_last')  # TF dimension ordering in this code


class Selection(Layer):
    def __init__(self, disparity_levels=None, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        if disparity_levels is None:
            disparity_levels = range(-16, 16, 1)

        super(Selection, self).__init__(**kwargs)

        self.disparity_levels = disparity_levels

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Selection` layer should be called '
                             'on a list of 2 inputs.')

    def call(self, inputs):

        # first we extract the left image from the original input
        image = inputs[0]
        # then the calculated disparity map that is the ouput of the Unet
        disparity_map = inputs[1]
        # initialize the stack of shifted left images
        shifted_images = []
        # loop over the different disparity levels and shift the left image accordingly, add it to the list
        for shift in self.disparity_levels:
            if shift > 0:
                shifted_images += [K.concatenate([image[..., shift:, :], K.zeros_like(image[..., :shift, :])], axis=2)]
            elif shift < 0:
                shifted_images += [K.concatenate([K.zeros_like(image[..., shift:, :]), image[..., :shift, :]], axis=2)]
            else:
                shifted_images += [image]

        # create a tensor of shape (None, im_rows, im_cols, disparity_levels)
        shifted_images_stack = K.stack(shifted_images)
        shifted_images_stack = K.permute_dimensions(shifted_images_stack, (1, 2, 3, 0, 4))

        # take the dot product with the disparity map along the disparity axis
        # and output the resulting right image of size (None, im_rows, im_cols)
        new_image = []
        for ch in range(3):
            new_image += [K.sum(shifted_images_stack[..., ch] * disparity_map, axis=3)]

        new_image = K.stack(new_image)
        new_image = K.permute_dimensions(new_image, (1, 2, 3, 0))

        return new_image

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Gradient(Layer):
    def __init__(self, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        super(Gradient, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, inputs):
        dinputs_dx_0 = inputs - K.concatenate([K.zeros_like(inputs[..., :1, :]), inputs[..., :-1, :]], axis=1)
        dinputs_dx_1 = inputs - K.concatenate([inputs[..., 1:, :], K.zeros_like(inputs[..., :1, :])], axis=1)

        dinputs_dy_0 = inputs - K.concatenate([K.zeros_like(inputs[..., :1]), inputs[..., :-1]], axis=2)
        dinputs_dy_1 = inputs - K.concatenate([inputs[..., 1:], K.zeros_like(inputs[..., :1])], axis=2)

        abs_gradient_sum = 0.25 * K.sqrt(
            K.square(dinputs_dx_0) + K.square(dinputs_dx_1) + K.square(dinputs_dy_0) + K.square(dinputs_dy_1))

        return abs_gradient_sum[..., 2:-2, 2:-2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 4, input_shape[2] - 4)


class Depth(Layer):
    def __init__(self, disparity_levels=None, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        if disparity_levels is None:
            disparity_levels = range(-3, 9, 1)

        # if none, initialize the disparity levels as described in deep3d
        super(Depth, self).__init__(**kwargs)

        self.disparity_levels = disparity_levels

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, disparity):

        depth = []
        for n, disp in enumerate(self.disparity_levels):
            depth += [disparity[..., n] * disp]

        depth = K.concatenate(depth, axis=0)
        return K.sum(depth, axis=0, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def get_unet(img_rows, img_cols, lr=1e-4):
    inputs = Input((img_rows, 2 * img_cols, 3))  # 2 channels: left and right images

    # split input left/right wise
    left_input_image = Lambda(lambda x: x[..., :img_cols, :])(inputs)
    right_input_image = Lambda(lambda x: x[..., img_cols:, :])(inputs)

    concatenated_images = concatenate([left_input_image, right_input_image], axis=3)

    conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(concatenated_images)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(rate=0.4)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(rate=0.4)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(rate=0.4)(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(rate=0.4)(conv9)

    # split into left/right disparity maps

    left_disparity_level_4 = Conv2DTranspose(32, (16, 16), strides=(16, 16), padding='same')(
        Lambda(lambda x: x[..., 128:])(pool4))
    right_disparity_level_4 = Conv2DTranspose(32, (16, 16), strides=(16, 16), padding='same')(
        Lambda(lambda x: x[..., :128])(pool4))

    left_disparity_level_3 = Conv2DTranspose(32, (8, 8), strides=(8, 8), padding='same')(
        Lambda(lambda x: x[..., 64:])(pool3))
    right_disparity_level_3 = Conv2DTranspose(32, (8, 8), strides=(8, 8), padding='same')(
        Lambda(lambda x: x[..., :64])(pool3))

    left_disparity_level_2 = Conv2DTranspose(32, (4, 4), strides=(4, 4), padding='same')(
        Lambda(lambda x: x[..., 32:])(pool2))
    right_disparity_level_2 = Conv2DTranspose(32, (4, 4), strides=(4, 4), padding='same')(
        Lambda(lambda x: x[..., :32])(pool2))

    left_disparity_level_1 = Lambda(lambda x: x[..., :32])(conv9)
    right_disparity_level_1 = Lambda(lambda x: x[..., 32:])(conv9)

    left_disparity = Lambda(lambda x: K.mean(K.stack([xi for xi in x]), axis=0))([left_disparity_level_1,
                                                                                  left_disparity_level_2,
                                                                                  left_disparity_level_3,
                                                                                  left_disparity_level_4])

    right_disparity = Lambda(lambda x: K.mean(K.stack([xi for xi in x]), axis=0))([right_disparity_level_1,
                                                                                   right_disparity_level_2,
                                                                                   right_disparity_level_3,
                                                                                   right_disparity_level_4])

    # use a softmax activation on the conv layer output to get a probabilistic disparity map
    left_disparity = SeparableConv2D(32, (3, 3), activation='softmax', padding='same')(left_disparity)

    right_disparity = SeparableConv2D(32, (3, 3), activation='softmax', padding='same')(right_disparity)

    left_disparity_levels = range(-16, 16, 1)
    right_reconstruct_im = Selection(disparity_levels=left_disparity_levels)([left_input_image, left_disparity])

    right_disparity_levels = range(16, -16, -1)
    left_reconstruct_im = Selection(disparity_levels=right_disparity_levels)([right_input_image, right_disparity])

    # concatenate left and right images along the channel axis
    output = concatenate([left_reconstruct_im, right_reconstruct_im], axis=2)

    # gradient regularization:
    depth_left = Depth(disparity_levels=left_disparity_levels)(left_disparity)
    depth_right = Depth(disparity_levels=left_disparity_levels)(right_disparity)
    depth_left_gradient = Gradient()(depth_left)
    depth_right_gradient = Gradient()(depth_right)

    left_input_im_gray = Lambda(lambda x: K.mean(x, axis=3))(left_input_image)
    right_input_im_gray = Lambda(lambda x: K.mean(x, axis=3))(right_input_image)

    left_input_im_gray_norm = Lambda(lambda x: x / K.max(x))(left_input_im_gray)
    right_input_im_gray_norm = Lambda(lambda x: x / K.max(x))(right_input_im_gray)

    image_left_gradient = Gradient()(left_input_im_gray_norm)
    image_right_gradient = Gradient()(right_input_im_gray_norm)

    weighted_gradient_left = Lambda(lambda x: x[0] * (1 - x[1]))([depth_left_gradient, image_left_gradient])
    weighted_gradient_right = Lambda(lambda x: x[0] * (1 - x[1]))([depth_right_gradient, image_right_gradient])

    model = Model(inputs=[inputs], outputs=[output, weighted_gradient_left, weighted_gradient_right])

    disp_map_model = Model(inputs=[inputs], outputs=[left_disparity, right_disparity])

    # we use L1 type loss as it has been shown to work better for that type of problem in the deep3d paper
    # (https://arxiv.org/abs/1604.03650)
    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error', loss_weights=[1., 0.001, 0.001])
    model.summary()

    return model, disp_map_model