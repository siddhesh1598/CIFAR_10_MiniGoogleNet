# import 
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from minigooglenet import MiniGoogleNet
from trainingmonitor import TrainingMonitor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import numpy as np
import argparse
import os

# initialize hyperparameters
EPOCHS = 70
INIT_LR = 5e-3

# learning rate decay function
def poly_decay(epoch):
	maxEpochs = EPOCHS
	baseLR = INIT_LR
	power = 1.0

	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	return alpha

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, 
	default="output",
	help="path to output folder")
ap.add_argument()

# load dataset
print("[INFO] loading data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# subtract mean from images
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels to integers
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct object for data augmentation
aug = ImageDataGenerator(
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
		fill_mode="nearest"
	)

# construct callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(
	os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(
	os.getpid())])
callbacks = [
	TrainingMonitor(figPath, 
		jsonPath=jsonPath),
	LearningRateScheduler(poly_decay)
]

# initialize the optimizer and the model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogleNet.build(width=32, height=32, depth=3, 
	classes=10)
model.compile(loss="categorical_crossentropy",
	optimizer=SGD, metrics=["accuracy"])

# train
print("[INFO] training model...")
model.fit_generator(
		aug.flow(trainX, trainY, batch_size=64),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // 64,
		epochs=EPOCHS,
		callbacks=callbacks,
		verbose=1
	)

# save the model
print("[INFO] saving model...")
model.save(os.path.sep.join([args["output"], "model.h5"]))


