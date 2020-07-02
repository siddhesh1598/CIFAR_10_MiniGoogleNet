# import
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):

	def __init__(self, figPath, jsonPath=None, startAt=0):
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath
		self.startAt = startAt

	def on_train_begin(self, logs={}):
		# init history dict
		self.H = {}

		# check if json file path is provided
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				# read the json file and store it in H
				self.H = json.loads(open(self.jsonPath).read())

				# check if starting point is provided
				if self.startAt > 0:
					# trim the entries past the starting point
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.startAt]

	def on_epoch_end(self, epoch, logs={}):
		# loop over the logs and update the loss/acc
		for (k, v) in logs.items():
			# get list of previous values 
			l = self.H.get(k, [])
			# appdend the current value to the list
			l.append(v)
			# update the list in the History dict
			self.H[k] = l

		# check if training history should be serialized 
		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")

		if len(self.H["loss"]) > 1:
			# plot training loss and acc
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			plt.plot(N, self.H["val_loss"], label="val_loss")
			plt.plot(N, self.H["accuracy"], label="train_accuracy")
			plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
			plt.title("Training Loss and Accuracy [Epoch {}]".format(
			len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()

			# save the figure
			plt.savefig(self.figPath)
			plt.close()
