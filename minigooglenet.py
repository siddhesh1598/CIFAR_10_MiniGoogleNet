# import
from tensorflow.keras.layers import (
	Conv2D,
	BatchNormalization,
	AveragePooling2D,
	MaxPooling2D,
	Flatten, 
	Dense,
	Dropout,
	Activation,
	Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow import keras as K

class MiniGoogleNet:

	@staticmethod
	def conv_module(x, K, kX, kY, stride, padding="same"):
		# CONV => RELU => BN
		x = Conv2D(K, (kX, kY), strides=stride, 
			padding=padding)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)

		return x

	@staticmethod
	def inception_module(x, numK_1x1, numK_3x3):
		conv_1x1 = MiniGoogleNet.conv_module(x, numK_1x1, 
			1, 1, (1, 1))
		conv_3x3 = MiniGoogleNet.conv_module(x, numK_3x3, 
			3, 3, (1, 1))
		x = Concatenate(axis=-1)([conv_1x1, conv_3x3])

		return x

	@staticmethod
	def downsample_module(x, K):
		conv_3x3 = MiniGoogleNet.conv_module(x, K, 
			3, 3, (2, 2), padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = Concatenate(axis=-1)([conv_3x3, pool])

		return x

	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)

		# INPUT => CONV
		inputs = Input(inputShape)
		x = MiniGoogleNet.conv_module(inputs, 96, 
			3, 3, (1, 1))

		# INCEPTION x 2 => DOWNSAMPLE
		x = MiniGoogleNet.inception_module(x, 32, 32)
		x = MiniGoogleNet.inception_module(x, 32, 48)
		x = MiniGoogleNet.downsample_module(x, 80)

		# INCEPTION x 4 => DOWNSAMPLE
		x = MiniGoogleNet.inception_module(x, 112, 48)
		x = MiniGoogleNet.inception_module(x, 96, 64)
		x = MiniGoogleNet.inception_module(x, 80, 80)
		x = MiniGoogleNet.inception_module(x, 48, 96)
		x = MiniGoogleNet.downsample_module(x, 96)

		# INCEPTION x 2 => AVGPOOL => DROPOUT
		x = MiniGoogleNet.inception_module(x, 176, 160)
		x = MiniGoogleNet.inception_module(x, 176, 160)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)

		# Softmax classifier
		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)

		# create the model
		model = Model(inputs, x, name="minigooglenet")

		return model

