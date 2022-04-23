from ctypes.wintypes import PINT
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow as tf
import sys
sys.path.append("./src")

class Involution(tf.keras.layers.Layer):
    def __init__(
        self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'channel': self.channel,
            'group_number': self.group_number,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'reduction_ratio': self.reduction_ratio
        })
        return config

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            tf.keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=self.kernel_size[0] * self.kernel_size[1] * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size[0] * self.kernel_size[1],
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size[0] * self.kernel_size[1],
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )
    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output

class Involution2D(layers.Layer):
    def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'same', dilation_rate = 1, groups = 1, reduce_ratio = 1):
        super(Involution2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.reduce_mapping = tf.keras.Sequential(
            [
                layers.Conv2D(filters // reduce_ratio, 1, padding = padding), 
                layers.BatchNormalization(), 
                layers.Activation('relu'), 
            ]
        )
        self.span_mapping = layers.Conv2D(kernel_size * kernel_size * groups, 1, padding = padding)
        self.initial_mapping = layers.Conv2D(self.filters, 1, padding = padding)
        if strides > 1:
            self.o_mapping = layers.AveragePooling2D(strides)
    
    def call(self, x):
        weight = self.span_mapping(self.reduce_mapping(x if self.strides == 1 else self.o_mapping(x)))
        _, h, w, c = K.int_shape(weight)
        weight = K.expand_dims(K.reshape(weight, (-1, h, w, self.groups, self.kernel_size * self.kernel_size)), axis = 4)
        out = tf.image.extract_patches(images = x if self.filters == c else self.initial_mapping(x),  
                                       sizes = [1, self.kernel_size, self.kernel_size, 1], 
                                       strides = [1, self.strides, self.strides, 1], 
                                       rates = [1, self.dilation_rate, self.dilation_rate, 1], 
                                       padding = "SAME" if self.padding == 'same' else "VALID")
        out = K.reshape(out, (-1, h, w, self.groups, self.filters // self.groups, self.kernel_size * self.kernel_size))
        out = K.sum(weight * out, axis = -1)
        out = K.reshape(out, (-1, h, w, self.filters))
        return out

class InRFNet:
    """Network design for InRFNet model"""
    @staticmethod
    def RFBlock(input_layer, conv_channels, padding="same", activation="relu"):
        # first stage
        layer_1 = Involution2D(conv_channels, kernel_size = 1, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(input_layer)
        layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
        layer_2 = tf.keras.layers.Activation(activation)(layer_2)
        layer_3 = Involution2D(conv_channels, kernel_size = 1, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(input_layer)
        layer_4 = tf.keras.layers.BatchNormalization()(layer_3)
        layer_4 = tf.keras.layers.Activation(activation)(layer_4)

        # second stage
        layer_5 = Involution2D(conv_channels, kernel_size = 1, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(input_layer)
        layer_6 = tf.keras.layers.BatchNormalization()(layer_5)
        layer_6 = tf.keras.layers.Activation(activation)(layer_6)
        layer_7 = Involution2D(conv_channels, kernel_size = 3, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(layer_2)
        layer_8 = tf.keras.layers.BatchNormalization()(layer_7)
        layer_8 = tf.keras.layers.Activation(activation)(layer_8)
        layer_9 = Involution2D(conv_channels, kernel_size = 5, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(layer_4)
        layer_10 = tf.keras.layers.BatchNormalization()(layer_9)
        layer_10 = tf.keras.layers.Activation(activation)(layer_10)
        
        # third stage
        layer_11 = Involution2D(conv_channels, kernel_size = 1, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(layer_6)
        layer_12 = tf.keras.layers.BatchNormalization()(layer_11)
        layer_12 = tf.keras.layers.Activation(activation)(layer_12)
        layer_13 = Involution2D(conv_channels, kernel_size = 3, strides = 1, padding = padding, dilation_rate = 4, groups = 1, reduce_ratio = 1)(layer_8)
        layer_14 = tf.keras.layers.BatchNormalization()(layer_13)
        layer_14 = tf.keras.layers.Activation(activation)(layer_14)
        layer_15 = Involution2D(conv_channels, kernel_size = 3, strides = 1, padding = padding, dilation_rate = 8, groups = 1, reduce_ratio = 1)(layer_10)
        layer_16 = tf.keras.layers.BatchNormalization()(layer_15)
        layer_16 = tf.keras.layers.Activation(activation)(layer_16)
        
        # fourth stage
        layer_17 = tf.keras.layers.concatenate([layer_12, layer_14, layer_16], axis = -1)

        # fifth stage
        layer_18 = Involution2D(conv_channels*3, kernel_size = 1, strides = 1, padding = padding, dilation_rate = 1, groups = 1, reduce_ratio = 1)(input_layer)
        # layer_18 = tf.keras.layers.Conv2D(conv_channels*3, kernel_size=(1, 1), padding = padding, dilation_rate=1)(input_layer)
        layer_19 = tf.keras.layers.BatchNormalization()(layer_18)
        layer_19 = tf.keras.layers.Activation(activation)(layer_19)

        layer_20 = tf.keras.layers.Add()([layer_17, layer_19])
        layer_21 = tf.keras.layers.Activation("relu")(layer_20)
        return layer_21

    @staticmethod
    def build(image_height, image_width, image_channel, number_classes):
        inputs = tf.keras.Input(shape = (image_height, image_width, image_channel))
        inv_1 = InRFNet.RFBlock(inputs, 8)
        inv_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inv_1)

        inv_2 = InRFNet.RFBlock(inv_1, 16)
        inv_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inv_2)
        
        inv_3 = InRFNet.RFBlock(inv_2, 16)
        inv_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inv_3)

        inv_4 = InRFNet.RFBlock(inv_3, 32)
        inv_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inv_4)

        inv_5 = InRFNet.RFBlock(inv_4, 64)
        inv_5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inv_5)

        x = tf.keras.layers.GlobalAveragePooling2D()(inv_5)
        outputs = tf.keras.layers.Dense(number_classes, activation = "softmax")(x)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="InRFNet_Model")
        return model