# All Includes

from src import read_dataset
import argparse
import pandas as pd
import time
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
#import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, CuDNNLSTM, concatenate, Flatten
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import subprocess

def freedsonComb98(counts, epSize, BM):
	if epSize != 60:
		CPM = 60/epSize*counts
	else:
		CPM = counts
	EE = []
	for i in range(len(counts)):
		if CPM[i] > 1951:
			EE.append(0.00094*CPM[i]+0.1346*BM-7.37418)*epSize/60
		else:
			EE.append(counts[i]*0.0000191*BM)
	return np.asarray(EE)


def freedsonAdult98(counts, epSize=2):
	# 2 sec epSize is default
	MET = []
	if epSize != 60:
		CPM = 60/epSize*counts
	else:
		CPM = counts

	for i in range(len(counts)):
		if CPM[i] < 50:
			MET.append(1.)
		else:
			MET.append(1.439008+0.000795*CPM[i])
#        elif (CPM[i] > 49) and (CPM[i] < 350):
#            MET.append(1.83)
#        elif (CPM[i] > 349) and (CPM[i] < 1200):
#           MET.append(1.935 + (0.003002 * CPM[i])) # R^2 = 0.74, SEE = 0.8
#        elif CPM[i] > 1199:
#            MET.append(2.768 + 0.0006397 * CPM[i]) # R^2 = 0.84, SEE = 0.9
	return np.asarray(MET)

def load_model(epSize):
	print("\nLoading model for epoch size: {}sec...".format(epSize))
	path = '/media/onur/OS_Install/Users/onur/Desktop/PhD/UML/RESEARCH/wearable-data/GUI/src'
	fullpath = path + '/' + 'theta_' + str(epSize) + 'sec.txt' # str format
	theta = []
	with open(fullpath) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			theta.append(row)

	theta = np.asarray(theta)
	return theta


def load_json_model(modelname):
	print("\nLoading the model from disk ...")
	json_file = open(modelname+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(modelname+'.h5')
	print("Loaded model from disk")
	return loaded_model


def one_hot(y_, n_classes=7):
	# Function to encode neural one-hot output labels from number indexes
	# e.g.:
	# one_hot(y_=[[5], [0], [3]], n_classes=6):
	#     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

	y_ = y_.reshape(len(y_))
	return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def get_fft_dataset(X_train, X_test, test_subject, epSize=2, Fs=100):
	# epSize = 2 sec by default
	# Fs = 100 Hz by default
	# Obtain FFT coefficients
	X_train_dfts = np.fft.fft(X_train, axis=1)
	X_train_dfts = abs(X_train_dfts[:,:int(epSize*Fs/2+1),:])
	X_test_dfts = np.fft.fft(X_test, axis=1)
	X_test_dfts = abs(X_test_dfts[:,:int(epSize*Fs/2+1),:])
	# Normalize them, get the highest=1. at least
	X_train_dfts = X_train_dfts/np.amax(abs(X_train_dfts))
	X_test_dfts = X_test_dfts/np.amax(abs(X_test_dfts))
	return X_train_dfts, X_test_dfts


def MAPE(y_true, y_pred):
	return np.mean(np.abs(y_true-y_pred)/y_true)

def MSE(y_true, y_pred):
	return (np.square(y_true - y_pred)).mean(axis=None)

def loadData(path_to_dataset, test_subject='S10', sensors=['A'], normalization=True, resample=50):
	### SENSOR PARAMETERS ###
	max_g = 8.0
	max_deg_per_sec = 2000.0
	max_uTesla = 4800.0

	[X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET] = read_dataset.main(path_to_dataset, test=test_subject)
	
	Xacc_train = X_train[:,:,:3]
	Xgyro_train = X_train[:,:,3:6]
	Xmag_train = X_train[:,:,6:]

	Xacc_test = X_test[:,:,:3]
	Xgyro_test = X_test[:,:,3:6]
	Xmag_test = X_test[:,:,6:]

	if normalization:
		Xacc_train = X_train[:,:,:3]/max_g
		Xgyro_train = X_train[:,:,3:6]/max_deg_per_sec
		Xmag_train = X_train[:,:,6:]/max_uTesla

		Xacc_test = X_test[:,:,:3]/max_g
		Xgyro_test = X_test[:,:,3:6]/max_deg_per_sec
		Xmag_test = X_test[:,:,6:]/max_uTesla

	X_train, X_test = None, None
	for sensor in sensors:
		if sensor == 'A' and X_train is None and X_test is None:
			X_train = Xacc_train
			X_test = Xacc_test
		elif sensor == 'A' and X_train is not None and X_test is not None:
			X_train = np.concatenate((X_train, Xacc_train), axis=2)
			X_test = np.concatenate((X_test, Xacc_test), axis=2)
			
		elif sensor == 'G' and X_train is not None and X_test is not None:
			X_train = np.concatenate((X_train, Xgyro_train), axis=2)
			X_test = np.concatenate((X_test, Xgyro_test), axis=2)
		elif sensor == 'G' and X_train is None and X_test is None:
			X_train = Xgyro_train
			X_test = Xgyro_test            

		elif sensor == 'M' and X_train is not None and X_test is not None:
			X_train = np.concatenate((X_train, Xmag_train), axis=2)
			X_test = np.concatenate((X_test, Xmag_test), axis=2)
		elif sensor == 'M' and X_train is None and X_test is None:
			X_train = Xmag_train
			X_test = Xmag_test
		else:
			ValueError("Incorrect sensor selection. Possible Selections: ['A', 'G', 'M']")


	# Downsample to 50 Hz
	if resample == 50:
		X_train = X_train[:,::2,:]
		X_test = X_test[:,::2,:]
	else:
		ValueError("Only resample to 50 Hz is available for now.")

	return [X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET]


def clean(directory):
	# moves model and loss figures inside the saved model folder
	if os.path.exists(directory):
		for img in os.listdir(os.getcwd()):
			if img.endswith('.png'):
				os.rename(img, directory+'/'+img)
	else:
		os.remove('model.png')
		os.remove('loss-acc.png')	


def shuffle_data(X_train, X_test, y_train, y_test):
	steps = X_train.shape[1]
	X = np.concatenate((X_train, X_test), axis=0)
	y = np.concatenate((y_train, y_test), axis=0)

	X_shuffled , y_shuffled = shuffle(X, y)
	X_tr_new , X_test_new, y_tr_new, y_test_new = train_test_split(X_shuffled, y_shuffled, test_size=.25)

	return X_tr_new , X_test_new, y_tr_new, y_test_new


def saveModel(directory, time_, learning_rate, decay_rate, dropout_rate, n_batch, n_epochs, lstm_hidden_units, fconn_units, loss_weights, lstm_reg, clf_reg, history):
	# Saving the model
	modelname = directory+"/model-{}".format(time_)
	# serialize model to JSON
	print("\nSaving the model ...")
	model_json = model.to_json()
	with open(modelname+".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(modelname+".h5")
	with open(modelname+'.txt', 'w') as file:
		file.write("Learning Rate: {} \n".format(learning_rate))
		file.write("Decay Rate: {} \n".format(decay_rate))
		file.write("Dropout Rate: {} \n".format(dropout_rate))
		file.write("Batch Size: {} \n".format(n_batch))
		file.write("# of Epochs: {} \n".format(n_epochs))
		file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
		file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
		file.write("Loss Weights: {} \n".format(loss_weights))
		file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
		file.write("Classification Regularization Coefficient : {} \n".format(clf_reg))        
		file.write("Train Classification Accuracy: {} \n".format(history.history['class_output_acc'][-1]))
		file.write("Test Classification Accuracy: {} \n".format(history.history['val_class_output_acc'][-1]))
		file.write("Train Count MSE: {} \n".format(history.history['count_output_loss'][-1]))
		file.write("Test Count MSE: {} \n".format(history.history['val_count_output_loss'][-1]))
	print("Model saved !")


def printCM(y_test, one_hot_predictions, n_classes, LABELS):
	print("")
	print("Precision: {:.4f}".format(metrics.precision_score(y_test, one_hot_predictions, average="weighted")))
	print("Recall: {:.4f}".format(metrics.recall_score(y_test, one_hot_predictions, average="weighted")))
	print("f1_score: {:.4f}".format(metrics.f1_score(y_test, one_hot_predictions, average="weighted")))
	for c in range(n_classes):
		print(LABELS[c], metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted"))
	mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
	print("mAP score: {:.4f}".format(mAP))

	print("")
	print("Confusion Matrix:")
	confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
	print(confusion_matrix)
	normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

	return confusion_matrix, normalised_confusion_matrix, mAP


def plotCM(normalised_confusion_matrix, confusion_matrix, test_subject, y_test, one_hot_predictions, mAP, n_classes, LABELS, directory, time_):
	# Plot Confusion Matrix:
	width = 8
	height = 8
	plt.figure(figsize=(width, height))
	plt.imshow(
		normalised_confusion_matrix,
		interpolation='nearest',
		cmap=plt.cm.Blues # Blues or grey
	)
	thresh = np.sum(confusion_matrix, axis=1)*.5
	for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
		if confusion_matrix[i, j] != 0:
			plt.text(j, i, format(confusion_matrix[i, j]),
				 horizontalalignment="center",
				 fontsize = 16,
				 color="white" if (confusion_matrix[i, j] > thresh[i]) and (confusion_matrix[i, j] != 0) else "black") # > if Blues, < if grey
	plt.title("{}-Confusion matrix (F1={:.4f} - mAP={:.4f}))".format(test_subject, metrics.f1_score(y_test[:,0], one_hot_predictions, average="weighted"), mAP), fontsize=18)
	#plt.colorbar()
	#plt.clim(0, np.amax(np.amax(confusion_matrix, axis=1)))
	tick_marks = np.arange(n_classes)
	plt.xticks(tick_marks, LABELS, rotation=55, horizontalalignment='right', fontsize=18)
	plt.yticks(tick_marks, LABELS, fontsize=18)
	plt.tight_layout()
	plt.ylabel('True label', rotation='vertical', fontsize=18)
	plt.xlabel('Predicted label', fontsize=18)
	plt.savefig(directory + '/' + time_+'_CM.png', bbox_inches='tight')


def plotLoss(history):
	# summarize history for accuracy
	for k, v in history.history.items():
		if k == 'acc' or k == 'class_output_acc':
			x1 = 'train accuracy'
			y1 = v
		elif k == 'val_acc' or k == 'val_class_output_acc':
			x2 = 'test accuracy'
			y2 = v
		elif k == 'loss':
			x3 = 'train loss'
			y3 = v
		elif k == 'val_loss':
			x4 = 'test loss'
			y4 = v
	plt.figure()  
	plt.plot(y1, 'r--')
	plt.plot(y2, 'b--')
	plt.title('model classification loss and accuracy')
	# summarize history for loss
	plt.plot(y3, 'r-')
	plt.plot(y4, 'b-')
	plt.ylabel('loss & accuracy (--)')
	plt.xlabel('epoch')
	plt.legend([x1, x2, x3, x4], loc='upper right')
	plt.savefig('loss-acc.png')


def plotPredictions(history, test_subject, test_kcal_MET, MET_predictions):
	# METs
	for k, v in history.history.items():
		if k == 'val_count_output_loss':
			x1 = 'Freedson98_Pred'
			y1 = v[-1]
		elif k == 'val_MET_output_loss':
			x2 = 'MET_Pred'
			y2 = v[-1]

	if x2: # MET is predicted directly
		x = x2
		y = y2
	else: # MET is not predicted directly by the model, i.e. counts are predicted
		x = x1
		y = y1

	plt.figure()  
	plt.title('MET Estimations on test subject ({}) - test-MSE={:.4f}'.format(test_subject, y))
	#plt.title('MET Estimations on test subject-{}- test-MAPE={:.4f}'.format(test_subject, MAPE(test_kcal_MET[:,1], MET_predictions)))
	plt.plot(test_kcal_MET[:,1], 'r', label='MET_True')
	plt.plot(MET_predictions, 'b*', label='MET_Pred')
	plt.ylabel('MET Rate')
	plt.xlabel('Epoch samples')
	plt.text(10. ,13. ,'LAYING')
	plt.text(210. ,13. ,'RUN')
	plt.text(360. ,13. ,'SIT')
	plt.text(470. ,13. ,'STAND')
	plt.text(640. ,13. ,'WALK')
	plt.text(780. ,13. ,'DOWNSTAIRS_UPSTAIRS')
	plt.axvline(x=150)
	plt.axvline(x=300)
	plt.axvline(x=450)
	plt.axvline(x=600)
	plt.axvline(x=750)
	plt.axvline(x=1050)
	plt.legend(['MET_True',x], loc='upper right')
	#plt.show(block=False)


def createModel(X_train, OUTPUT, n_classes, LSTM_layers, LSTM_Units, lstm_reg, FC_layers, FC_Units, clf_reg, dropout_rate=0):
	# Model Definition
	OUTPUTS = []
	
	raw_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
	if LSTM_layers == 1:
		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False,
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
		if dropout_rate != 0:
			xlstm = Dropout(dropout_rate)(xlstm)        
	else:   
		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=True,
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
		if dropout_rate != 0:
			xlstm = Dropout(dropout_rate)(xlstm)    

		for i in range(1, LSTM_layers-1):
			xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=True,
						kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
						recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
						bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
						activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)    
			if dropout_rate != 0:
				xlstm = Dropout(dropout_rate)(xlstm)

		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False,
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)
		if dropout_rate != 0:
			xlstm = Dropout(dropout_rate)(xlstm)

	if 'activity_classification' in OUTPUT: 
		class_predictions = Dense(n_classes, activation='softmax', 
					kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
					bias_regularizer=tf.keras.regularizers.l2(clf_reg),
					activity_regularizer=tf.keras.regularizers.l1(clf_reg),
					name='class_output')(xlstm)
		OUTPUTS.append(class_predictions)

	if 'count_estimations' in OUTPUT: 
		if FC_layers == 0:
			count_estimations = Dense(3, activation='relu',
					name='count_output')(xlstm)
		else:  
			if 'MET_estimations' not in OUTPUT:         
				for i in range(FC_layers):
					xlstm = Dense(fconn_units, activation='relu')(xlstm)
					if dropout_rate != 0:
						xlstm = Dropout(dropout_rate)(xlstm)
				count_estimations = Dense(3, activation='relu',
						name='count_output')(xlstm)
			else:
				for i in range(FC_layers):
					xlstm = Dense(fconn_units, activation='relu')(xlstm)
					if dropout_rate != 0:
						xlstm = Dropout(dropout_rate)(xlstm)
				count_estimations = Dense(3, activation='relu',
						name='count_output')(xlstm)
				MET_estimations = Dense(1, activation='relu',
						name='MET_output')(xlstm) 		
		OUTPUTS.append(count_estimations)

	if 'MET_estimations' in OUTPUT:
		if FC_layers == 0:
			MET_estimations = Dense(1, activation='relu',
					name='MET_output')(xlstm)
		else:
			if 'count_estimations' not in OUTPUT:
				for i in range(FC_layers):
					xlstm = Dense(fconn_units, activation='relu')(xlstm)
					if dropout_rate != 0:
						xlstm = Dropout(dropout_rate)(xlstm)                
				MET_estimations = Dense(1, activation='relu',
						name='MET_output')(xlstm)        

		OUTPUTS.append(MET_estimations)     

	model = Model(inputs=raw_inputs, outputs=OUTPUTS)

	return model
		

path_to_dataset= './data' 

# Output classes to learn how to classify
LABELS = [
	"SITTING",
	"LAYING",
	"STANDING",
	"WALKING",
	"RUNNING",
	"WALKING_DOWNSTAIRS",
	"WALKING_UPSTAIRS",
]

def MET_AG_main(test_subject='S10', shuffle=False, hyperparameters = [], ftune = False, modelname = '', saveModel = True):
	global directory
	# Load "X" and "y" and "epochs" (the neural network's training and testing inputs)
	[X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET] = loadData(path_to_dataset, test_subject=test_subject, sensors=['A', 'G'], normalization=True, resample=50)
	if shuffle:
		X_train, X_test, y_train, y_test = shuffle_data(X_train, X_test, y_train, y_test)
		test_subject = "shuffled"
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print(epochs.shape)
	print(test_epochs.shape)
	print(kcal_MET.shape)
	print(test_kcal_MET.shape)

	# Input Data

	# 7770 training series (with 0% overlap)
	# 630 testing series
	# 200 timesteps per series
	# 9 input parameters per timestep

	n_classes = int(max(y_train.max(), y_test.max())+1) # Total classes (should go up, or should go down)
	print("\nn_classes = {}".format(n_classes))
	
	# Default Training Hyperparameters
	learning_rate = 2e-4
	decay_rate = 1e-5
	dropout_rate = 0.5
	n_batch = 300
	n_epochs = 1500  # Loop 500 times on the dataset
	LSTM_layers = 2
	lstm_hidden_units = 32
	FC_layers = 1
	fconn_units = 32
	loss_weights = [1., .001]
	lstm_reg = 2e-4
	clf_reg = 2e-4

	# Use input hyperparameters:
	if hyperparameters:
		learning_rate = hyperparameters[0]
		decay_rate = hyperparameters[1]
		dropout_rate = hyperparameters[2]
		n_batch = hyperparameters[3]
		n_epochs = hyperparameters[4]  # Loop 5000 times on the dataset
		LSTM_layers = hyperparameters[5]
		lstm_hidden_units = hyperparameters[6]
		FC_layers = hyperparameters[7]
		fconn_units = hyperparameters[8]
		loss_weights = hyperparameters[9]
		lstm_reg = hyperparameters[10]
		clf_reg = hyperparameters[11]

	# Some debugging info

	print("\nSome useful info to get an insight on dataset's shape and normalisation:")
	print("(X shape, y shape, epoch shape, every X's mean, every X's standard deviation)")
	print(X_test.shape, y_test.shape, test_epochs.shape, np.mean(X_test), np.std(X_test))
	print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

	if ftune:
		# Load Model
		model = load_json_model(modelname)
	else:
		print("\nTraining the model from scratch...")
		model = createModel(X_train, ['activity_classification', 'MET_estimations'], n_classes, LSTM_layers, lstm_hidden_units, lstm_reg, FC_layers, fconn_units, clf_reg, dropout_rate)

	print(model.summary()) # summarize layers
	plot_model(model, to_file='recurrent_neural_network.png') # plot graph
	model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
		loss_weights=loss_weights,
		metrics=['accuracy'])

	# Train the model
	history=model.fit(X_train, [one_hot(y_train, n_classes), kcal_MET[:,1]], batch_size=n_batch, epochs=n_epochs, validation_data=(X_test, [one_hot(y_test, n_classes), test_kcal_MET[:,1]]))
	#history=model.fit([X_train, X_train_dfts], [one_hot(y_train, n_classes), epochs], batch_size=n_batch, epochs=n_epochs, validation_data=([X_test, X_test_dfts], [one_hot(y_test, n_classes), test_epochs]))
	print("Optimization Finished!")
	predictions = model.predict(X_test)
	MET_predictions = predictions[1]
	#predictions = model.predict([X_test, X_test_dfts])

	# summarize history for accuracy
	plotLoss(history)
	
	# METs
	plotPredictions(history, test_subject, test_kcal_MET, MET_predictions)


	### CONFUSION MATRIX AND METRICS ###
	one_hot_predictions = predictions[0].argmax(1)

	# Print Confusion Matrix and calculate mAP, F1
	confusion_matrix, normalised_confusion_matrix, mAP = printCM(y_test, one_hot_predictions, n_classes, LABELS)

	# SAVE THE MODEL ?
	if saveModel:
		time_ = time.strftime("%Y%m%d-%H%M%S")
		directory = os.getcwd() + '/saved_models/' + test_subject + '_' + time_
		if not os.path.exists(directory):
			os.makedirs(directory)

		plt.savefig(directory + '/' +time_+'.png')
		
		# Plot and save Confusion Matrix
		plotCM(normalised_confusion_matrix, confusion_matrix, test_subject, y_test, one_hot_predictions, mAP, n_classes, LABELS, directory, time_)

		# Saving the model
		modelname = directory+"/model-{}".format(time_)
		# serialize model to JSON
		print("\nSaving the model ...")
		model_json = model.to_json()
		with open(modelname+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(modelname+".h5")
		with open(modelname+'.txt', 'w') as file:
			file.write("Learning Rate: {} \n".format(learning_rate))
			file.write("Decay Rate: {} \n".format(decay_rate))
			file.write("Dropout Rate: {} \n".format(dropout_rate))
			file.write("Batch Size: {} \n".format(n_batch))
			file.write("# of Epochs: {} \n".format(n_epochs))
			file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
			file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
			file.write("Loss Weights: {} \n".format(loss_weights))
			file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
			file.write("Classification Regularization Coefficient : {} \n".format(clf_reg)) 
			for k, v in history.history.items():
				file.write("{}: {:.4f} \n".format(k, v[-1]))       

		print("Model saved !")		

		#submission
		int_y_test = [int(i) for i in y_test]
		int_pred = [int(i) for i in one_hot_predictions]
		
		# Calculate MAE for each class
		sit_indices = [i for i, x in enumerate(int_y_test) if x == 0]
		stand_indices = [i for i, x in enumerate(int_y_test) if x == 1]
		lying_indices = [i for i, x in enumerate(int_y_test) if x == 2]
		walking_indices = [i for i, x in enumerate(int_y_test) if x == 3]
		walking_ds_indices = [i for i, x in enumerate(int_y_test) if x == 4]
		walking_us_indices = [i for i, x in enumerate(int_y_test) if x == 5]
		running_indices = [i for i, x in enumerate(int_y_test) if x == 6]

		sit_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in sit_indices]
		sit_MAE = sum(sit_AE)/len(sit_AE)
		sit_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in sit_indices]
		sit_MAPE = sum(sit_APE)/len(sit_APE)
		stand_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in stand_indices]
		stand_MAE = sum(stand_AE)/len(stand_AE)
		stand_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in stand_indices]
		stand_MAPE = sum(stand_APE)/len(stand_APE)
		lying_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in lying_indices]
		lying_MAE = sum(lying_AE)/len(lying_AE)        
		lying_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in lying_indices]
		lying_MAPE = sum(lying_APE)/len(lying_APE)
		walking_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_indices]
		walking_MAE = sum(walking_AE)/len(walking_AE)  
		walking_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in walking_indices]
		walking_MAPE = sum(walking_APE)/len(walking_APE)
		walking_ds_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_ds_indices]
		walking_ds_MAE = sum(walking_ds_AE)/len(walking_ds_AE)  
		walking_ds_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in walking_ds_indices]
		walking_ds_MAPE = sum(walking_ds_APE)/len(walking_ds_APE)
		walking_us_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_us_indices]
		walking_us_MAE = sum(walking_us_AE)/len(walking_us_AE)           
		walking_us_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in walking_us_indices]
		walking_us_MAPE = sum(walking_us_APE)/len(walking_us_APE)
		running_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in running_indices]
		running_MAE = sum(running_AE)/len(running_AE)  
		running_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in running_indices]
		running_MAPE = sum(running_APE)/len(running_APE)

		MAE = 1/n_classes*(sit_MAE+stand_MAE+lying_MAE+walking_MAE+walking_ds_MAE+walking_us_MAE+running_MAE)
		MAPE = 1/n_classes*(sit_MAPE+stand_MAPE+lying_MAPE+walking_MAPE+walking_ds_MAPE+walking_us_MAPE+running_MAPE)

		submission = pd.DataFrame({
			"Class": [LABELS[i] for i in int_y_test],
			"MET": test_kcal_MET[:,1],
			"Predicted_Class": [LABELS[i] for i in int_pred],
			"Predicted_MET": MET_predictions.reshape((MET_predictions.shape[0],))

			})

		submission.to_csv(directory+'/{}-result.csv'.format(test_subject), index=False)
		with open(modelname+'-results.csv', 'w') as file:
			file.write("MAE:,{}\nMAPE:,{}\n".format(MAE[0], MAPE[0]))
			file.write("SITTING_MAE:,{}\nSITTING_MAPE:,{}\n".format(sit_MAE[0], sit_MAPE[0]))
			file.write("STANDING_MAE:,{}\nSTANDING_MAPE:,{}\n".format(stand_MAE, stand_MAPE))
			file.write("LYING_MAE:,{}\nLYING_MAPE:,{}\n".format(lying_MAE[0],lying_MAPE[0]))
			file.write("WALKING_MAE:,{}\nWALKING_MAPE:,{}\n".format(walking_MAE[0], walking_MAPE[0]))
			file.write("WALKING_DS_MAE:,{}\nWALKING_DS_MAPE:,{}\n".format(walking_ds_MAE[0], walking_ds_MAPE[0]))
			file.write("WALKING_US_MAE:,{}\nWALKING_US_MAPE:,{}\n".format(walking_us_MAE[0], walking_us_MAPE[0]))
			file.write("RUNNING_MAE:,{}\nRUNNING_MAPE:,{}\n".format(running_MAE[0], running_MAPE[0]))
			file.write("SITTING_AE,SITTING_APE,STANDING_AE,STANDING_APE,LYING_AE,LYING_APE,WALKING_AE,WALKING_APE,WALKING_DOWNSTAIRS_AE,WALKING_DOWNSTAIRS_APE,WALKING_UPSTAIRS_AE,WALKING_UPSTAIRS_APE,RUNNING_AE,RUNNING_APE\n")
			for i in range(len(sit_AE)):
				file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(sit_AE[i][0], sit_APE[i][0], stand_AE[i][0], stand_APE[i][0], lying_AE[i][0], lying_APE[i][0], walking_AE[i][0], walking_APE[i][0], walking_ds_AE[i][0], walking_ds_APE[i][0], walking_us_AE[i][0], walking_us_APE[i][0], running_AE[i][0], running_APE[i][0]))




	else:
		print("Model not saved.")

def count_MET_main(test_subject='S10', shuffle=False, hyperparameters = [], ftune = False, modelname = '', saveModel = True):
	global directory
	# Load "X" and "y" and "epochs" (the neural network's training and testing inputs)
	
	[X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET] = loadData(path_to_dataset, test_subject=test_subject, sensors=['A'], normalization=True, resample=50)
	if shuffle:
		X_train, X_test, y_train, y_test = shuffle_data(X_train, X_test, y_train, y_test)
		test_subject = "shuffled"
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print(epochs.shape)
	print(test_epochs.shape)
	print(kcal_MET.shape)
	print(test_kcal_MET.shape)

	# Input Data

	# 7770 training series (with 0% overlap)
	# 630 testing series
	# 200 timesteps per series
	# 9 input parameters per timestep

	n_classes = int(max(y_train.max(), y_test.max())+1) # Total classes (should go up, or should go down)
	print("\nn_classes = {}".format(n_classes))
	
	# Default Training Hyperparameters
	learning_rate = 2e-4
	decay_rate = 1e-5
	dropout_rate = 0.5
	n_batch = 300
	n_epochs = 1500  # Loop 500 times on the dataset
	LSTM_layers = 2
	lstm_hidden_units = 32
	FC_layers = 1
	fconn_units = 32
	loss_weights = [1., .001, .1]
	lstm_reg = 2e-4
	clf_reg = 2e-4

	# Use input hyperparameters:
	if hyperparameters:
		learning_rate = hyperparameters[0]
		decay_rate = hyperparameters[1]
		dropout_rate = hyperparameters[2]
		n_batch = hyperparameters[3]
		n_epochs = hyperparameters[4]  # Loop 5000 times on the dataset
		LSTM_layers = hyperparameters[5]
		lstm_hidden_units = hyperparameters[6]
		FC_layers = hyperparameters[7]
		fconn_units = hyperparameters[8]
		#loss_weights = hyperparameters[9]
		lstm_reg = hyperparameters[10]
		clf_reg = hyperparameters[11]

	# Some debugging info

	print("\nSome useful info to get an insight on dataset's shape and normalisation:")
	print("(X shape, y shape, epoch shape, every X's mean, every X's standard deviation)")
	print(X_test.shape, y_test.shape, test_epochs.shape, np.mean(X_test), np.std(X_test))
	print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

	if ftune:
		# Load Model
		model = load_json_model(modelname)
	else:
		print("\nTraining the model from scratch...")

		# Model Definition
		OUTPUTS = ['activity_classification', 'count_estimations', 'MET_estimations']
		model = createModel(X_train=X_train, OUTPUT=OUTPUTS, 
					n_classes=n_classes, LSTM_layers=1, LSTM_Units=lstm_hidden_units, 
					lstm_reg=lstm_reg, FC_layers=1, FC_Units=fconn_units, 
					clf_reg=clf_reg, dropout_rate=0.5)
		
	print(model.summary()) # summarize layers
	plot_model(model, to_file='recurrent_neural_network.png') # plot graph
	model.compile(loss=['categorical_crossentropy', 'mean_squared_error', 'mean_squared_error'],
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
		loss_weights=loss_weights,
		metrics=['accuracy'])

	# Train the model
	history=model.fit(X_train, [one_hot(y_train), epochs, kcal_MET[:,1]], batch_size=n_batch, epochs=n_epochs, validation_data=(X_test, [one_hot(y_test), test_epochs, test_kcal_MET[:,1]]))
	#history=model.fit([X_train, X_train_dfts], [one_hot(y_train), epochs], batch_size=n_batch, epochs=n_epochs, validation_data=([X_test, X_test_dfts], [one_hot(y_test), test_epochs]))
	print("Optimization Finished!")
	predictions = model.predict(X_test)
	#predictions = model.predict([X_test, X_test_dfts])

	# summarize history for accuracy
	plt.figure()  
	plt.plot(history.history['class_output_acc'], 'r')
	plt.plot(history.history['val_class_output_acc'], 'b')
	plt.title('model classification accuracy')
	plt.legend(['train accuracy', 'validation accuracy'], loc='upper right')
	plt.savefig('acc.png')

	# summarize history for loss
	plt.figure()
	plt.plot(history.history['loss'], 'r')
	plt.plot(history.history['val_loss'], 'b')
	plt.ylabel('weighted loss')
	plt.xlabel('epoch')
	plt.legend(['train loss', 'validation loss'], loc='upper right')
	plt.savefig('weighted_loss.png')

	# plot individual losses
	plt.figure()
	plt.plot(history.history['class_output_loss'], 'r')
	plt.plot(history.history['val_class_output_loss'], 'b')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train loss', 'validation loss'], loc='upper right')
	plt.savefig('clf_loss.png')
	plt.figure()
	plt.plot(history.history['count_output_loss'], 'r')
	plt.plot(history.history['val_count_output_loss'], 'b')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train loss', 'validation loss'], loc='upper right')
	plt.savefig('count_loss.png')
	plt.figure()
	plt.plot(history.history['MET_output_loss'], 'r')
	plt.plot(history.history['val_MET_output_loss'], 'b')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train loss', 'validation loss'], loc='upper right')
	plt.savefig('MET_loss.png')
	# METs
	count_predictions = predictions[1]
	MET_predictions = predictions[2]
	plt.figure()  
	test_count2MET = freedsonAdult98(predictions[1][:,0])
	plt.title('MET Estimations on test subject-{}-\nval_MET-MSE={:.4f} val_count-MSE={:.4f}'.format(test_subject, history.history['val_MET_output_loss'][-1], history.history['val_count_output_loss'][-1]))
	#plt.title('MET Estimations on test subject-{}- test-MAPE={:.4f}'.format(test_subject, MAPE(test_kcal_MET[:,1], MET_predictions)))
	plt.plot(test_kcal_MET[:,1], 'ro', label='MET_GT')
	plt.plot(MET_predictions, 'b*', label='MET_pred')
	plt.plot(test_count2MET, 'kx', label='Freedson98_pred')
	plt.ylabel('MET Rate')
	plt.xlabel('Epoch samples')
	plt.text(10. ,13. ,'LAYING')
	plt.text(210. ,13. ,'RUN')
	plt.text(360. ,13. ,'SIT')
	plt.text(470. ,13. ,'STAND')
	plt.text(640. ,13. ,'WALK')
	plt.text(780. ,13. ,'DOWNSTAIRS_UPSTAIRS')
	plt.axvline(x=150)
	plt.axvline(x=300)
	plt.axvline(x=450)
	plt.axvline(x=600)
	plt.axvline(x=750)
	plt.axvline(x=1050)
	plt.legend(['MET_GT','MET_pred({:.4f})'.format(MSE(test_kcal_MET[:,1], MET_predictions)), 'Freedson98_pred({:.4f})'.format(MSE(test_kcal_MET[:,1], test_count2MET))], loc='upper right')

	### CONFUSION MATRIX AND METRICS ###

	# Results

	one_hot_predictions = predictions[0].argmax(1)

	print("")
	print("Precision: {:.4f}".format(metrics.precision_score(y_test, one_hot_predictions, average="weighted")))
	print("Recall: {:.4f}".format(metrics.recall_score(y_test, one_hot_predictions, average="weighted")))
	print("f1_score: {:.4f}".format(metrics.f1_score(y_test, one_hot_predictions, average="weighted")))
	for c in range(n_classes):
		print(LABELS[c], metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted"))
	mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
	print("mAP score: {:.4f}".format(mAP))

	print("")
	print("Confusion Matrix:")
	confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
	print(confusion_matrix)
	normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

	# SAVE THE MODEL ?

	if saveModel:
		time_ = time.strftime("%Y%m%d-%H%M%S")
		directory = os.getcwd() + '/saved_models/' + test_subject + '_' + time_
		if not os.path.exists(directory):
			os.makedirs(directory)

		plt.savefig(directory + '/' +time_+'.png')
		# Plot Confusion Matrix:
		plotCM(normalised_confusion_matrix, confusion_matrix, test_subject, y_test, one_hot_predictions, mAP, n_classes, LABELS, directory, time_)

		# Saving the model
		modelname = directory+"/model-{}".format(time_)
		# serialize model to JSON
		print("\nSaving the model ...")
		model_json = model.to_json()
		with open(modelname+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(modelname+".h5")
		with open(modelname+'.txt', 'w') as file:
			file.write("Learning Rate: {} \n".format(learning_rate))
			file.write("Decay Rate: {} \n".format(decay_rate))
			file.write("Dropout Rate: {} \n".format(dropout_rate))
			file.write("Batch Size: {} \n".format(n_batch))
			file.write("# of Epochs: {} \n".format(n_epochs))
			file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
			file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
			file.write("Loss Weights: {} \n".format(loss_weights))
			file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
			file.write("Classification Regularization Coefficient : {} \n".format(clf_reg))
			for k, v in history.history.items():
				file.write("{}: {:.4f} \n".format(k, v[-1]))
		print("Model saved !")
		
		#submission
		int_y_test = [int(i) for i in y_test]
		int_pred = [int(i) for i in one_hot_predictions]
		
		# Calculate MAE for each class
		sit_indices = [i for i, x in enumerate(int_y_test) if x == 0]
		stand_indices = [i for i, x in enumerate(int_y_test) if x == 1]
		lying_indices = [i for i, x in enumerate(int_y_test) if x == 2]
		walking_indices = [i for i, x in enumerate(int_y_test) if x == 3]
		walking_ds_indices = [i for i, x in enumerate(int_y_test) if x == 4]
		walking_us_indices = [i for i, x in enumerate(int_y_test) if x == 5]
		running_indices = [i for i, x in enumerate(int_y_test) if x == 6]

		for i in range(2): # first iteration for MET pred, second for count2MET pred
			name = 'MET'
			if i == 1:
				name = 'CPM'
				MET_predictions = test_count2MET

			sit_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in sit_indices]
			sit_MAE = sum(sit_AE)/len(sit_AE)
			sit_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in sit_indices]
			sit_MAPE = sum(sit_APE)/len(sit_APE)
			stand_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in stand_indices]
			stand_MAE = sum(stand_AE)/len(stand_AE)
			stand_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in stand_indices]
			stand_MAPE = sum(stand_APE)/len(stand_APE)
			lying_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in lying_indices]
			lying_MAE = sum(lying_AE)/len(lying_AE)        
			lying_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in lying_indices]
			lying_MAPE = sum(lying_APE)/len(lying_APE)
			walking_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_indices]
			walking_MAE = sum(walking_AE)/len(walking_AE)  
			walking_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in walking_indices]
			walking_MAPE = sum(walking_APE)/len(walking_APE)
			walking_ds_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_ds_indices]
			walking_ds_MAE = sum(walking_ds_AE)/len(walking_ds_AE)  
			walking_ds_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in walking_ds_indices]
			walking_ds_MAPE = sum(walking_ds_APE)/len(walking_ds_APE)
			walking_us_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_us_indices]
			walking_us_MAE = sum(walking_us_AE)/len(walking_us_AE)           
			walking_us_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in walking_us_indices]
			walking_us_MAPE = sum(walking_us_APE)/len(walking_us_APE)
			running_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in running_indices]
			running_MAE = sum(running_AE)/len(running_AE)  
			running_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1] for i in running_indices]
			running_MAPE = sum(running_APE)/len(running_APE)

			MAE = 1/n_classes*(sit_MAE+stand_MAE+lying_MAE+walking_MAE+walking_ds_MAE+walking_us_MAE+running_MAE)
			MAPE = 1/n_classes*(sit_MAPE+stand_MAPE+lying_MAPE+walking_MAPE+walking_ds_MAPE+walking_us_MAPE+running_MAPE)

			submission = pd.DataFrame({
				"Class": [LABELS[i] for i in int_y_test],
				"MET": test_kcal_MET[:,1],
				"Predicted_Class": [LABELS[i] for i in int_pred],
				"Predicted_MET": MET_predictions.reshape((MET_predictions.shape[0],))

				})

			submission.to_csv(directory+'/{}-{}-result.csv'.format(test_subject, name), index=False)
			with open(modelname+'_'+name+'-results.csv', 'w') as file:
				file.write("MAE:,{}\nMAPE:,{}\n".format(MAE, MAPE))
				file.write("SITTING_MAE:,{}\nSITTING_MAPE:,{}\n".format(sit_MAE, sit_MAPE))
				file.write("STANDING_MAE:,{}\nSTANDING_MAPE:,{}\n".format(stand_MAE, stand_MAPE))
				file.write("LYING_MAE:,{}\nLYING_MAPE:,{}\n".format(lying_MAE,lying_MAPE))
				file.write("WALKING_MAE:,{}\nWALKING_MAPE:,{}\n".format(walking_MAE, walking_MAPE))
				file.write("WALKING_DS_MAE:,{}\nWALKING_DS_MAPE:,{}\n".format(walking_ds_MAE, walking_ds_MAPE))
				file.write("WALKING_US_MAE:,{}\nWALKING_US_MAPE:,{}\n".format(walking_us_MAE, walking_us_MAPE))
				file.write("RUNNING_MAE:,{}\nRUNNING_MAPE:,{}\n".format(running_MAE, running_MAPE))
				file.write("SITTING_AE,SITTING_APE,STANDING_AE,STANDING_APE,LYING_AE,LYING_APE,WALKING_AE,WALKING_APE,WALKING_DOWNSTAIRS_AE,WALKING_DOWNSTAIRS_APE,WALKING_UPSTAIRS_AE,WALKING_UPSTAIRS_APE,RUNNING_AE,RUNNING_APE\n")
				for i in range(len(sit_AE)):
					file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(sit_AE[i], sit_APE[i], stand_AE[i], stand_APE[i], lying_AE[i], lying_APE[i], walking_AE[i], walking_APE[i], walking_ds_AE[i], walking_ds_APE[i], walking_us_AE[i], walking_us_APE[i], running_AE[i], running_APE[i]))
	else:
		print("Model not saved.")

def count_main(test_subject='S10', shuffle=False, hyperparameters = [], ftune = False, modelname = '', saveModel = True):
	global directory
	# Load "X" and "y" and "epochs" (the neural network's training and testing inputs)
	
	[X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET] = loadData(path_to_dataset, test_subject=test_subject, sensors=['A', 'G'], normalization=True, resample=50)
	if shuffle:
		X_train, X_test, y_train, y_test = shuffle_data(X_train, X_test, y_train, y_test)
		test_subject = "shuffled"
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print(epochs.shape)
	print(test_epochs.shape)
	print(kcal_MET.shape)
	print(test_kcal_MET.shape)

	# Input Data

	# 7770 training series (with 0% overlap)
	# 630 testing series
	# 200 timesteps per series
	# 9 input parameters per timestep

	n_classes = int(max(y_train.max(), y_test.max())+1) # Total classes (should go up, or should go down)
	print("\nn_classes = {}".format(n_classes))
	
	# Default Training Hyperparameters
	learning_rate = 2e-4
	decay_rate = 1e-5
	dropout_rate = 0.5
	n_batch = 300
	n_epochs = 1500  # Loop 500 times on the dataset
	lstm_hidden_units = 32
	fconn_units = 32
	loss_weights = [1., .001]
	lstm_reg = 2e-4
	clf_reg = 2e-4

	# Use input hyperparameters:
	if hyperparameters:
		learning_rate = hyperparameters[0]
		decay_rate = hyperparameters[1]
		dropout_rate = hyperparameters[2]
		n_batch = hyperparameters[3]
		n_epochs = hyperparameters[4]  # Loop 5000 times on the dataset
		LSTM_layers = hyperparameters[5]
		lstm_hidden_units = hyperparameters[6]
		FC_layers = hyperparameters[7]
		fconn_units = hyperparameters[8]
		#loss_weights = hyperparameters[9]
		lstm_reg = hyperparameters[10]
		clf_reg = hyperparameters[11]

	# Some debugging info

	print("\nSome useful info to get an insight on dataset's shape and normalisation:")
	print("(X shape, y shape, epoch shape, every X's mean, every X's standard deviation)")
	print(X_test.shape, y_test.shape, test_epochs.shape, np.mean(X_test), np.std(X_test))
	print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

	if ftune:
		# Load Model
		model = load_json_model(modelname)
	else:
		print("\nTraining the model from scratch...")

		# Model Definition
		raw_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False, # True if other LSTM follows
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
		xlstm = Dropout(dropout_rate)(xlstm)
		#xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False,
		#            kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
		#            recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
		#            bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
		#            activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)
		#xlstm = Dropout(dropout_rate)(xlstm)
		"""
		xlstm = CuDNNLSTM(lstm_hidden_units,
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)
		
		dft_inputs = Input(shape=(X_train_dfts.shape[1], X_train_dfts.shape[2]))
		dftdense = CuDNNLSTM(lstm_hidden_units,
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(dft_inputs)
		merged = concatenate([xlstm, dftdense])
		merged = Dropout(dropout_rate)(merged)
		"""

		class_predictions = Dense(n_classes, activation='softmax', 
					kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
					bias_regularizer=tf.keras.regularizers.l2(clf_reg),
					activity_regularizer=tf.keras.regularizers.l1(clf_reg),
					name='class_output')(xlstm)
		
		xcount = Dense(fconn_units, activation='relu')(xlstm)
		xcount = Dropout(dropout_rate)(xcount)
		count_estimations = Dense(3, activation='relu',
					name='count_output')(xcount)

		#model = Model(inputs=[raw_inputs, dft_inputs], outputs=[class_predictions, count_estimations])
		model = Model(inputs=raw_inputs, outputs=[class_predictions, count_estimations])

	print(model.summary()) # summarize layers
	plot_model(model, to_file='recurrent_neural_network.png') # plot graph
	model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
		loss_weights=loss_weights,
		metrics=['accuracy'])

	# Train the model
	history=model.fit(X_train, [one_hot(y_train), epochs], batch_size=n_batch, epochs=n_epochs, validation_data=(X_test, [one_hot(y_test), test_epochs]))
	#history=model.fit([X_train, X_train_dfts], [one_hot(y_train), epochs], batch_size=n_batch, epochs=n_epochs, validation_data=([X_test, X_test_dfts], [one_hot(y_test), test_epochs]))
	print("Optimization Finished!")
	predictions = model.predict(X_test)
	#predictions = model.predict([X_test, X_test_dfts])

	# summarize history for accuracy
	plt.figure()  
	plt.plot(history.history['class_output_acc'], 'r--')
	plt.plot(history.history['val_class_output_acc'], 'b--')
	plt.title('model classification loss and accuracy')
	# summarize history for loss
	plt.plot(history.history['loss'], 'r-')
	plt.plot(history.history['val_loss'], 'b-')
	plt.ylabel('loss & accuracy (--)')
	plt.xlabel('epoch')
	plt.legend(['train accuracy', 'test accuracy', 'train loss', 'test loss'], loc='upper right')
	plt.savefig('loss-acc.png')
	# METs
	MET_predictions = freedsonAdult98(predictions[1][:,0])
	plt.figure()  
	plt.title('MET Estimations on test subject-{}- MAE:0.3436 - MAPE:11.35'.format(test_subject))
	plt.plot(test_kcal_MET[:,1], 'r', label='MET_GT')
	plt.plot(MET_predictions, 'b*', label='Freedson98_pred')
	plt.ylabel('MET Rate')
	plt.xlabel('Epoch samples')
	plt.text(20. ,9.2 ,'SIT')
	plt.text(165. ,9.2 ,'STAND')
	plt.text(325. ,9.2 ,'WALK')
	plt.text(485. ,9.2 ,'RUN')
	plt.text(615. ,7.2 ,'DOWNSTAIRS &\nUPSTAIRS')
	plt.text(920. ,7.2 ,'LAYING')
	plt.axvline(x=150)
	plt.axvline(x=300)
	plt.axvline(x=450)
	plt.axvline(x=600)
	plt.axvline(x=900)
	plt.axvline(x=1050)
	plt.legend(['MET_GT', 'Freedson98_pred'], loc='upper right')
	#plt.show(block=False)

	### CONFUSION MATRIX AND METRICS ###

	# Results

	one_hot_predictions = predictions[0].argmax(1)

	print("")
	print("Precision: {:.4f}".format(metrics.precision_score(y_test, one_hot_predictions, average="weighted")))
	print("Recall: {:.4f}".format(metrics.recall_score(y_test, one_hot_predictions, average="weighted")))
	print("f1_score: {:.4f}".format(metrics.f1_score(y_test, one_hot_predictions, average="weighted")))
	for c in range(n_classes):
		print(LABELS[c], metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted"))
	mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
	print("mAP score: {:.4f}".format(mAP))

	print("")
	print("Confusion Matrix:")
	confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
	print(confusion_matrix)
	normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

	# SAVE THE MODEL ?

	if saveModel:
		time_ = time.strftime("%Y%m%d-%H%M%S")
		directory = os.getcwd() + '/saved_models/' + test_subject + '_' + time_
		if not os.path.exists(directory):
			os.makedirs(directory)

		plt.savefig(directory + '/' +time_+'.png')

		# Plot and save Confusion Matrix
		plotCM(normalised_confusion_matrix, confusion_matrix, test_subject, y_test, one_hot_predictions, mAP, n_classes, LABELS, directory, time_)


		# Saving the model
		modelname = directory+"/model-{}".format(time_)
		# serialize model to JSON
		print("\nSaving the model ...")
		model_json = model.to_json()
		with open(modelname+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(modelname+".h5")
		with open(modelname+'.txt', 'w') as file:
			file.write("Learning Rate: {} \n".format(learning_rate))
			file.write("Decay Rate: {} \n".format(decay_rate))
			file.write("Dropout Rate: {} \n".format(dropout_rate))
			file.write("Batch Size: {} \n".format(n_batch))
			file.write("# of Epochs: {} \n".format(n_epochs))
			file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
			file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
			file.write("Loss Weights: {} \n".format(loss_weights))
			file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
			file.write("Classification Regularization Coefficient : {} \n".format(clf_reg))        
			file.write("Train Classification Accuracy: {} \n".format(history.history['class_output_acc'][-1]))
			file.write("Test Classification Accuracy: {} \n".format(history.history['val_class_output_acc'][-1]))
			file.write("Train Count MSE: {} \n".format(history.history['count_output_loss'][-1]))
			file.write("Test Count MSE: {} \n".format(history.history['val_count_output_loss'][-1]))
		print("Model saved !")
		
		#submission
		int_y_test = [int(i) for i in y_test]
		int_pred = [int(i) for i in one_hot_predictions]
		
		# Calculate MAE for each class
		sit_indices = [i for i, x in enumerate(int_y_test) if x == 0]
		stand_indices = [i for i, x in enumerate(int_y_test) if x == 1]
		lying_indices = [i for i, x in enumerate(int_y_test) if x == 2]
		walking_indices = [i for i, x in enumerate(int_y_test) if x == 3]
		walking_ds_indices = [i for i, x in enumerate(int_y_test) if x == 4]
		walking_us_indices = [i for i, x in enumerate(int_y_test) if x == 5]
		running_indices = [i for i, x in enumerate(int_y_test) if x == 6]

		sit_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in sit_indices]
		sit_MAE = sum(sit_AE)/len(sit_AE)
		sit_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in sit_indices]
		sit_MAPE = sum(sit_APE)/len(sit_APE)
		stand_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in stand_indices]
		stand_MAE = sum(stand_AE)/len(stand_AE)
		stand_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in stand_indices]
		stand_MAPE = sum(stand_APE)/len(stand_APE)
		lying_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in lying_indices]
		lying_MAE = sum(lying_AE)/len(lying_AE)        
		lying_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in lying_indices]
		lying_MAPE = sum(lying_APE)/len(lying_APE)
		walking_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_indices]
		walking_MAE = sum(walking_AE)/len(walking_AE)  
		walking_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in walking_indices]
		walking_MAPE = sum(walking_APE)/len(walking_APE)
		walking_ds_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_ds_indices]
		walking_ds_MAE = sum(walking_ds_AE)/len(walking_ds_AE)  
		walking_ds_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in walking_ds_indices]
		walking_ds_MAPE = sum(walking_ds_APE)/len(walking_ds_APE)
		walking_us_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_us_indices]
		walking_us_MAE = sum(walking_us_AE)/len(walking_us_AE)           
		walking_us_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in walking_us_indices]
		walking_us_MAPE = sum(walking_us_APE)/len(walking_us_APE)
		running_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in running_indices]
		running_MAE = sum(running_AE)/len(running_AE)  
		running_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in running_indices]
		running_MAPE = sum(running_APE)/len(running_APE)

		MAE = 1/n_classes*(sit_MAE+stand_MAE+lying_MAE+walking_MAE+walking_ds_MAE+walking_us_MAE+running_MAE)
		MAPE = 1/n_classes*(sit_MAPE+stand_MAPE+lying_MAPE+walking_MAPE+walking_ds_MAPE+walking_us_MAPE+running_MAPE)

		submission = pd.DataFrame({
			"Class": [LABELS[i] for i in int_y_test],
			"MET": test_kcal_MET[:,1],
			"Predicted_Class": [LABELS[i] for i in int_pred],
			"Predicted_MET": MET_predictions.reshape((MET_predictions.shape[0],))

			})
		"""
		"MAE": MAE,
		"MAPE": MAPE,
		"SITTING_AE": sit_AE,
		"SITTING_MAE": sit_MAE,
		"SITTING_APE": sit_APE,
		"SITTING_MAPE": sit_MAPE,
		"STANDING_AE": stand_AE,
		"STANDING_MAE": stand_MAE,
		"STANDING_APE": stand_APE,
		"STANDING_MAPE": stand_MAPE,
		"LYING_AE": lying_AE,
		"LYING_MAE": lying_MAE,
		"LYING_APE": lying_APE,
		"LYING_MAPE": lying_MAPE,
		"WALKING_AE": walking_AE,
		"WALKING_MAE": walking_MAE,
		"WALKING_APE": walking_APE,
		"WALKING_MAPE": walking_MAPE,
		"WALKING_DOWNSTAIRS_AE": walking_ds_AE,
		"WALKING_DOWNSTAIRS_MAE": walking_ds_MAE,
		"WALKING_DOWNSTAIRS_APE": walking_ds_APE,
		"WALKING_DOWNSTAIRS_MAPE": walking_ds_MAPE,
		"WALKING_UPSTAIRS_AE": walking_us_AE,
		"WALKING_UPSTAIRS_MAE": walking_us_MAE,
		"WALKING_UPSTAIRS_APE": walking_us_APE,
		"WALKING_UPSTAIRS_MAPE": walking_us_MAPE,
		"RUNNING_AE": running_AE,
		"RUNNING_MAE": running_MAE,
		"RUNNING_APE": running_APE,
		"RUNNING_MAPE": running_MAPE
		"""
		submission.to_csv(directory+'/{}-result.csv'.format(test_subject), index=False)
		with open(modelname+'-results.csv', 'w') as file:
			file.write("MAE:,{}\nMAPE:,{}\n".format(MAE, MAPE))
			file.write("SITTING_MAE:,{}\nSITTING_MAPE:,{}\n".format(sit_MAE, sit_MAPE))
			file.write("STANDING_MAE:,{}\nSTANDING_MAPE:,{}\n".format(stand_MAE, stand_MAPE))
			file.write("LYING_MAE:,{}\nLYING_MAPE:,{}\n".format(lying_MAE,lying_MAPE))
			file.write("WALKING_MAE:,{}\nWALKING_MAPE:,{}\n".format(walking_MAE, walking_MAPE))
			file.write("WALKING_DS_MAE:,{}\nWALKING_DS_MAPE:,{}\n".format(walking_ds_MAE, walking_ds_MAPE))
			file.write("WALKING_US_MAE:,{}\nWALKING_US_MAPE:,{}\n".format(walking_us_MAE, walking_us_MAPE))
			file.write("RUNNING_MAE:,{}\nRUNNING_MAPE:,{}\n".format(running_MAE, running_MAPE))
			file.write("SITTING_AE,SITTING_APE,STANDING_AE,STANDING_APE,LYING_AE,LYING_APE,WALKING_AE,WALKING_APE,WALKING_DOWNSTAIRS_AE,WALKING_DOWNSTAIRS_APE,WALKING_UPSTAIRS_AE,WALKING_UPSTAIRS_APE,RUNNING_AE,RUNNING_APE\n")
			for i in range(len(sit_AE)):
				file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(sit_AE[i], sit_APE[i], stand_AE[i], stand_APE[i], lying_AE[i], lying_APE[i], walking_AE[i], walking_APE[i], walking_ds_AE[i], walking_ds_APE[i], walking_us_AE[i], walking_us_APE[i], running_AE[i], running_APE[i]))



	else:
		print("Model not saved.")

def clfOnly(test_subject='S10', shuffle=False, hyperparameters = [], ftune = False, modelname = '', saveModel = True):
	global directory
	# Load "X" and "y" and "epochs" (the neural network's training and testing inputs)
	
	[X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET] = loadData(path_to_dataset, test_subject=test_subject, sensors=['A', 'G'], normalization=True, resample=50)
	if shuffle:
		X_train, X_test, y_train, y_test = shuffle_data(X_train, X_test, y_train, y_test)
		test_subject = "shuffled"
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print(epochs.shape)
	print(test_epochs.shape)
	print(kcal_MET.shape)
	print(test_kcal_MET.shape)

	# Input Data

	# 7770 training series (with 0% overlap)
	# 630 testing series
	# 200 timesteps per series
	# 9 input parameters per timestep

	n_classes = int(max(y_train.max(), y_test.max())+1) # Total classes (should go up, or should go down)
	print("\nn_classes = {}".format(n_classes))
	
	# Default Training Hyperparameters
	learning_rate = 2e-4
	decay_rate = 1e-5
	dropout_rate = 0.5
	n_batch = 300
	n_epochs = 1500  # Loop 500 times on the dataset
	lstm_hidden_units = 32
	fconn_units = 32
	loss_weights = [1., .001]
	lstm_reg = 2e-4
	clf_reg = 2e-4

	# Use input hyperparameters:
	if hyperparameters:
		learning_rate = hyperparameters[0]
		decay_rate = hyperparameters[1]
		dropout_rate = hyperparameters[2]
		n_batch = hyperparameters[3]
		n_epochs = hyperparameters[4]  # Loop 500 times on the dataset
		lstm_hidden_units = hyperparameters[5]
		fconn_units = hyperparameters[6]
		loss_weights = hyperparameters[7]
		lstm_reg = hyperparameters[8]
		clf_reg = hyperparameters[9]

	# Some debugging info

	print("\nSome useful info to get an insight on dataset's shape and normalisation:")
	print("(X shape, y shape, epoch shape, every X's mean, every X's standard deviation)")
	print(X_test.shape, y_test.shape, test_epochs.shape, np.mean(X_test), np.std(X_test))
	print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

	if ftune:
		# Load Model
		model = load_json_model(modelname)
	else:
		print("\nTraining the model from scratch...")

		# Model Definition
		raw_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False, #True if another LSTM layer follows
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
		xlstm = Dropout(dropout_rate)(xlstm)
		#xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False,
		#            kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
		#            recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
		#            bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
		#            activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)
		#xlstm = Dropout(dropout_rate)(xlstm)

		class_predictions = Dense(n_classes, activation='softmax', 
					kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
					bias_regularizer=tf.keras.regularizers.l2(clf_reg),
					activity_regularizer=tf.keras.regularizers.l1(clf_reg),
					name='class_output')(xlstm)


		model = Model(inputs=raw_inputs, outputs=class_predictions)

	print(model.summary()) # summarize layers
	plot_model(model, to_file='recurrent_neural_network.png') # plot graph
	model.compile(loss='categorical_crossentropy',
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
		metrics=['accuracy'])

	# Train the model
	history=model.fit(X_train, one_hot(y_train, n_classes), batch_size=n_batch, epochs=n_epochs, validation_data=(X_test, one_hot(y_test, n_classes)))
	print("Optimization Finished!")
	predictions = model.predict(X_test)

	# summarize history for accuracy
	plt.figure()  
	plt.plot(history.history['acc'], 'r--')
	plt.plot(history.history['val_acc'], 'b--')
	plt.title('model classification loss and accuracy')
	# summarize history for loss
	plt.plot(history.history['loss'], 'r-')
	plt.plot(history.history['val_loss'], 'b-')
	plt.ylabel('loss & accuracy (--)')
	plt.xlabel('epoch')
	plt.legend(['train accuracy', 'test accuracy', 'train loss', 'test loss'], loc='upper right')
	plt.savefig('loss-acc.png')
	plt.show(block=False)

	### CONFUSION MATRIX AND METRICS ###

	# Results

	one_hot_predictions = predictions.argmax(1)

	print("")
	print("Precision: {:.4f}".format(metrics.precision_score(y_test, one_hot_predictions, average="weighted")))
	print("Recall: {:.4f}".format(metrics.recall_score(y_test, one_hot_predictions, average="weighted")))
	print("f1_score: {:.4f}".format(metrics.f1_score(y_test, one_hot_predictions, average="weighted")))
	for c in range(n_classes):
		print(LABELS[c], metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted"))
	mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test[:,0], n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
	print("mAP score: {:.4f}".format(mAP))

	print("")
	print("Confusion Matrix:")
	confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
	print(confusion_matrix)
	normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

	# SAVE THE MODEL ?

	if saveModel:
		time_ = time.strftime("%Y%m%d-%H%M%S")
		directory = os.getcwd() + '/saved_models/' + test_subject + '_' + time_
		if not os.path.exists(directory):
			os.makedirs(directory)

		plt.savefig(directory + '/' +time_+'.png')
		# Plot Confusion Matrix:
		width = 8
		height = 8
		plt.figure(figsize=(width, height))
		plt.imshow(
			normalised_confusion_matrix,
			interpolation='nearest',
			cmap=plt.cm.Blues # Blues or grey
		)
		thresh = confusion_matrix.max() *.5
		for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
			plt.text(j, i, format(confusion_matrix[i, j]),
					 horizontalalignment="center",
					 color="white" if (confusion_matrix[i, j] > thresh) and (confusion_matrix[i, j] != 0) else "black") # > if Blues, < if grey
		plt.title("{}-Confusion matrix (F1={:.4f} - mAP={:.4f}) \n(normalised to % of total test data)".format(test_subject, metrics.f1_score(y_test, one_hot_predictions, average="weighted"), mAP))
		plt.colorbar()
		tick_marks = np.arange(n_classes)
		plt.xticks(tick_marks, LABELS, rotation=90)
		plt.yticks(tick_marks, LABELS)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(directory + '/' + time_+'_CM.png')

		# Saving the model
		modelname = directory+"/model-{}".format(time_)
		# serialize model to JSON
		print("\nSaving the model ...")
		model_json = model.to_json()
		with open(modelname+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(modelname+".h5")
		with open(modelname+'.txt', 'w') as file:
			file.write("Learning Rate: {} \n".format(learning_rate))
			file.write("Decay Rate: {} \n".format(decay_rate))
			file.write("Dropout Rate: {} \n".format(dropout_rate))
			file.write("Batch Size: {} \n".format(n_batch))
			file.write("# of Epochs: {} \n".format(n_epochs))
			file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
			file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
			file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
			file.write("Classification Regularization Coefficient : {} \n".format(clf_reg))        
			file.write("Train Classification Accuracy: {} \n".format(history.history['acc'][-1]))
			file.write("Test Classification Accuracy: {} \n".format(history.history['val_acc'][-1]))
		print("Model saved !")


	else:
		print("Model not saved.")

def countOnly(test_subject='S10', shuffle=False, hyperparameters = [], ftune = False, modelname = '', saveModel = True):
	global directory
	# Load "X" and "y" and "epochs" (the neural network's training and testing inputs)
	
	[X_train, y_train, epochs, X_test, y_test, test_epochs, kcal_MET, test_kcal_MET] = loadData(path_to_dataset, test_subject=test_subject, sensors=['A', 'G'], normalization=True, resample=50)
	if shuffle:
		X_train, X_test, y_train, y_test = shuffle_data(X_train, X_test, y_train, y_test)
		test_subject = "shuffled"
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print(epochs.shape)
	print(test_epochs.shape)
	print(kcal_MET.shape)
	print(test_kcal_MET.shape)

	# Input Data

	# 7770 training series (with 0% overlap)
	# 630 testing series
	# 200 timesteps per series
	# 9 input parameters per timestep

	n_classes = int(max(y_train.max(), y_test.max())+1) # Total classes (should go up, or should go down)
	print("\nn_classes = {}".format(n_classes))
	
	# Default Training Hyperparameters
	learning_rate = 2e-4
	decay_rate = 1e-5
	dropout_rate = 0.5
	n_batch = 300
	n_epochs = 1500  # Loop 500 times on the dataset
	lstm_hidden_units = 32
	fconn_units = 32
	loss_weights = [1., .001]
	lstm_reg = 2e-4
	clf_reg = 2e-4

	# Use input hyperparameters:
	if hyperparameters:
		learning_rate = hyperparameters[0]
		decay_rate = hyperparameters[1]
		dropout_rate = hyperparameters[2]
		n_batch = hyperparameters[3]
		n_epochs = hyperparameters[4]  # Loop 500 times on the dataset
		lstm_hidden_units = hyperparameters[5]
		fconn_units = hyperparameters[6]
		loss_weights = hyperparameters[7]
		lstm_reg = hyperparameters[8]
		clf_reg = hyperparameters[9]


	# Some debugging info

	print("\nSome useful info to get an insight on dataset's shape and normalisation:")
	print("(X shape, y shape, epoch shape, every X's mean, every X's standard deviation)")
	print(X_test.shape, y_test.shape, test_epochs.shape, np.mean(X_test), np.std(X_test))
	print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

	if ftune:
		# Load Model
		model = load_json_model(modelname)
	else:
		print("\nTraining the model from scratch...")

		# Model Definition
		raw_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=True, #True if another LSTM layer follows
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
		xlstm = Dropout(dropout_rate)(xlstm)
		xlstm = CuDNNLSTM(lstm_hidden_units, return_sequences=False,
					kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
					recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
					bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
					activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)
		xlstm = Dropout(dropout_rate)(xlstm)
		
		#xcount = Dense(fconn_units, activation='relu')(xlstm)
		#count = Dropout(dropout_rate)(xcount)
		
		count_estimations = Dense(epochs.shape[1], activation='relu', 
					name='class_output')(xlstm) # count of Dense used, xlstm if NoFc


		model = Model(inputs=raw_inputs, outputs=count_estimations)

	print(model.summary()) # summarize layers
	plot_model(model, to_file='recurrent_neural_network.png') # plot graph
	model.compile(loss='mean_squared_error',
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
		metrics=['accuracy'])

	# Train the model
	history=model.fit(X_train, epochs, batch_size=n_batch, epochs=n_epochs, validation_data=(X_test, test_epochs))
	print("Optimization Finished!")
	predictions = model.predict(X_test)

	# summarize history for accuracy
	plt.figure()  
	plt.plot(history.history['acc'], 'r--')
	plt.plot(history.history['val_acc'], 'b--')
	plt.title('model classification loss and accuracy')
	# summarize history for loss
	plt.plot(history.history['loss'], 'r-')
	plt.plot(history.history['val_loss'], 'b-')
	plt.ylabel('loss & accuracy (--)')
	plt.xlabel('epoch')
	plt.legend(['train accuracy', 'test accuracy', 'train loss', 'test loss'], loc='upper right')
	plt.savefig('loss-acc.png')
	#plt.show(block=False)
	# METs
	MET_predictions = freedsonAdult98(predictions[:,0])
	plt.figure()  
	plt.title('MET Estimations on test subject-{}- MAE:0.2885 - MAPE:9.18'.format(test_subject))
	#plt.title('MET Estimations on test subject-{}- test-MAPE={:.4f}'.format(test_subject, MAPE(test_kcal_MET[:,1], MET_predictions)))
	plt.plot(test_kcal_MET[:,1], 'r', label='MET_GT')
	plt.plot(MET_predictions, 'b*', label='Freedson98_pred')
	plt.ylabel('MET Rate')
	plt.xlabel('Epoch samples')
	plt.text(20. ,9.2 ,'SIT')
	plt.text(165. ,9.2 ,'STAND')
	plt.text(325. ,9.2 ,'WALK')
	plt.text(485. ,9.2 ,'RUN')
	plt.text(615. ,7.2 ,'DOWNSTAIRS &\nUPSTAIRS')
	plt.text(920. ,7.2 ,'LAYING')
	plt.axvline(x=150)
	plt.axvline(x=300)
	plt.axvline(x=450)
	plt.axvline(x=600)
	plt.axvline(x=900)
	plt.axvline(x=1050)
	plt.legend(['MET_GT', 'Freedson98_pred'], loc='upper right')
	#plt.show(block=False)

	# SAVE THE MODEL ?

	if saveModel:
		time_ = time.strftime("%Y%m%d-%H%M%S")
		directory = os.getcwd() + '/saved_models/' + test_subject + '_' + time_
		if not os.path.exists(directory):
			os.makedirs(directory)

		plt.savefig(directory + '/' +time_+'.png')

		# Saving the model
		modelname = directory+"/model-{}".format(time_)
		# serialize model to JSON
		print("\nSaving the model ...")
		model_json = model.to_json()
		with open(modelname+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(modelname+".h5")
		with open(modelname+'.txt', 'w') as file:
			file.write("Learning Rate: {} \n".format(learning_rate))
			file.write("Decay Rate: {} \n".format(decay_rate))
			file.write("Dropout Rate: {} \n".format(dropout_rate))
			file.write("Batch Size: {} \n".format(n_batch))
			file.write("# of Epochs: {} \n".format(n_epochs))
			file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
			file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
			file.write("Loss Weights: {} \n".format(loss_weights))
			file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
			file.write("Classification Regularization Coefficient : {} \n".format(clf_reg))        
			file.write("Train Count MSE: {} \n".format(history.history['loss'][-1]))
			file.write("Test Count MSE: {} \n".format(history.history['val_loss'][-1]))
		print("Model saved !")
		
		#submission
		int_y_test = [int(i) for i in y_test]
		
		# Calculate MAE for each class
		sit_indices = [i for i, x in enumerate(int_y_test) if x == 0]
		stand_indices = [i for i, x in enumerate(int_y_test) if x == 1]
		lying_indices = [i for i, x in enumerate(int_y_test) if x == 2]
		walking_indices = [i for i, x in enumerate(int_y_test) if x == 3]
		walking_ds_indices = [i for i, x in enumerate(int_y_test) if x == 4]
		walking_us_indices = [i for i, x in enumerate(int_y_test) if x == 5]
		running_indices = [i for i, x in enumerate(int_y_test) if x == 6]

		sit_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in sit_indices]
		sit_MAE = sum(sit_AE)/len(sit_AE)
		sit_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in sit_indices]
		sit_MAPE = sum(sit_APE)/len(sit_APE)
		stand_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in stand_indices]
		stand_MAE = sum(stand_AE)/len(stand_AE)
		stand_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in stand_indices]
		stand_MAPE = sum(stand_APE)/len(stand_APE)
		lying_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in lying_indices]
		lying_MAE = sum(lying_AE)/len(lying_AE)        
		lying_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in lying_indices]
		lying_MAPE = sum(lying_APE)/len(lying_APE)
		walking_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_indices]
		walking_MAE = sum(walking_AE)/len(walking_AE)  
		walking_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in walking_indices]
		walking_MAPE = sum(walking_APE)/len(walking_APE)
		walking_ds_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_ds_indices]
		walking_ds_MAE = sum(walking_ds_AE)/len(walking_ds_AE)  
		walking_ds_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in walking_ds_indices]
		walking_ds_MAPE = sum(walking_ds_APE)/len(walking_ds_APE)
		walking_us_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in walking_us_indices]
		walking_us_MAE = sum(walking_us_AE)/len(walking_us_AE)           
		walking_us_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in walking_us_indices]
		walking_us_MAPE = sum(walking_us_APE)/len(walking_us_APE)
		running_AE = [abs(test_kcal_MET[i,1] - MET_predictions[i]) for i in running_indices]
		running_MAE = sum(running_AE)/len(running_AE)  
		running_APE = [abs(test_kcal_MET[i,1] - MET_predictions[i])/test_kcal_MET[i,1]*100 for i in running_indices]
		running_MAPE = sum(running_APE)/len(running_APE)

		MAE = 1/n_classes*(sit_MAE+stand_MAE+lying_MAE+walking_MAE+walking_ds_MAE+walking_us_MAE+running_MAE)
		MAPE = 1/n_classes*(sit_MAPE+stand_MAPE+lying_MAPE+walking_MAPE+walking_ds_MAPE+walking_us_MAPE+running_MAPE)

		submission = pd.DataFrame({
			"Class": [LABELS[i] for i in int_y_test],
			"MET": test_kcal_MET[:,1],
			"Predicted_MET": MET_predictions.reshape((MET_predictions.shape[0],))

			})

		submission.to_csv(directory+'/{}-result.csv'.format(test_subject), index=False)
		with open(modelname+'-results.csv', 'w') as file:
			file.write("MAE:,{}\nMAPE:,{}\n".format(MAE, MAPE))
			file.write("SITTING_MAE:,{}\nSITTING_MAPE:,{}\n".format(sit_MAE, sit_MAPE))
			file.write("STANDING_MAE:,{}\nSTANDING_MAPE:,{}\n".format(stand_MAE, stand_MAPE))
			file.write("LYING_MAE:,{}\nLYING_MAPE:,{}\n".format(lying_MAE,lying_MAPE))
			file.write("WALKING_MAE:,{}\nWALKING_MAPE:,{}\n".format(walking_MAE, walking_MAPE))
			file.write("WALKING_DS_MAE:,{}\nWALKING_DS_MAPE:,{}\n".format(walking_ds_MAE, walking_ds_MAPE))
			file.write("WALKING_US_MAE:,{}\nWALKING_US_MAPE:,{}\n".format(walking_us_MAE, walking_us_MAPE))
			file.write("RUNNING_MAE:,{}\nRUNNING_MAPE:,{}\n".format(running_MAE, running_MAPE))
			file.write("SITTING_AE,SITTING_APE,STANDING_AE,STANDING_APE,LYING_AE,LYING_APE,WALKING_AE,WALKING_APE,WALKING_DOWNSTAIRS_AE,WALKING_DOWNSTAIRS_APE,WALKING_UPSTAIRS_AE,WALKING_UPSTAIRS_APE,RUNNING_AE,RUNNING_APE\n")
			for i in range(len(sit_AE)):
				file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(sit_AE[i], sit_APE[i], stand_AE[i], stand_APE[i], lying_AE[i], lying_APE[i], walking_AE[i], walking_APE[i], walking_ds_AE[i], walking_ds_APE[i], walking_us_AE[i], walking_us_APE[i], running_AE[i], running_APE[i]))

	else:
		print("Model not saved.")


if __name__ == '__main__':
	global workDir
	global directory # saveDir

	# Default Training Hyperparameters
	learning_rate = 2e-4
	decay_rate = 1e-5
	dropout_rate = 0.5
	n_batch = 300
	n_epochs = 1500  # Loop 500 times on the dataset
	LSTM_layers = 2
	lstm_hidden_units = 32
	FC_layers = 1
	fconn_units = 32
	loss_weights = [1., .001]
	lstm_reg = 2e-4
	clf_reg = 2e-4

	## setup parser
	parser = argparse.ArgumentParser(description="Train HAR", add_help=True)
	parser.add_argument('--function', action="store", help="select which model to train")
	parser.add_argument('--testSubject', action="store", help="select which subject is going to be tested.. '0': 10-fold cv")
	parser.add_argument('--fineTune', action="store", help="True if fineTune")
	parser.add_argument('--modelname', action="store", help="specify model file")
	parser.add_argument('--saveModel', action="store", help="True if model is to be saved")
	parser.add_argument('--shuffle', action="store", help="True if shuffle the dataset")
	parser.add_argument('--lr', action="store", help="learning rate (2e-4)")
	parser.add_argument('--dr', action="store", help="decay rate (1e-5)")
	parser.add_argument('--drop', action="store", help="dropout rate (.5)")
	parser.add_argument('--batch', action="store", help="n_batch (300)")
	parser.add_argument('--epoch', action="store", help="n_epochs (1500)")
	parser.add_argument('--lstmLayers', action="store", help="# of lstm layers (2)")
	parser.add_argument('--lstmUnits', action="store", help="#of units in lstm layer (32)")
	parser.add_argument('--fcLayers', action="store", help="#of fc layers (1)")
	parser.add_argument('--fcUnits', action="store", help="#of units in fc layer (32)")
	parser.add_argument('--lossWeights', action="store", help="loss_weights for multitask training ([1., .001])")
	parser.add_argument('--lstmReg', action="store", help="lstm layer regularization coef (2e-4)")
	parser.add_argument('--clfReg', action="store", help="classifier layer regularization coef (2e-4)")

	args = parser.parse_args()

	if args.modelname:
		modelname = args.modelname
	if args.lr:
		learning_rate = float(args.lr)
	if args.dr:
		decay_rate = float(args.dr)
	if args.drop:
		dropout_rate = float(args.drop)
	if args.batch:
		n_batch = int(args.batch)
	if args.epoch:
		n_epochs = int(args.epoch)
	if args.lstmLayers:
		LSTM_layers = int(args.lstmLayers)
	if args.lstmUnits:
		lstm_hidden_units = int(args.lstmUnits)
	if args.fcLayers:
		FC_layers = int(args.fcLayers)
	if args.fcUnits:
		fconn_units = int(args.fcUnits)
	if args.lossWeights:
		loss_weights = [1., float(args.lossWeights)]
	if args.lstmReg:
		lstm_reg = float(args.lstmReg)
	if args.clfReg:
		clf_reg = float(args.clfReg)

	hyperparameters = [learning_rate, decay_rate, dropout_rate, n_batch, n_epochs, LSTM_layers, lstm_hidden_units, FC_layers, fconn_units, loss_weights, lstm_reg, clf_reg] 

	if args.function == 'clfOnly':
		if args.shuffle:
			test_subject = 'S10' #dummy

			for i in range(0,3):
				# train 4 times:
				clfOnly(test_subject=test_subject, shuffle=True, hyperparameters=hyperparameters)
				clean(directory)

		elif args.testSubject == '0' or args.testSubject is None: #10-fold cv
			testList = ['S1', 'S2', 'S3', 'S4', 'S5',
						'S6', 'S7', 'S8', 'S9', 'S10']

			for test_subject in testList:
				clfOnly(test_subject=test_subject, hyperparameters=hyperparameters)
				clean(directory)

		elif int(args.testSubject) <= 10 and int(args.testSubject) >= 1:
			clfOnly(test_subject='S'+args.testSubject, hyperparameters=hyperparameters, ftune=args.fineTune, modelname=args.modelname, saveModel=args.saveModel)
			clean(directory)

	elif args.function == 'MET_AG_main':  
		if args.shuffle:
			test_subject = 'S10' #dummy
			
			for i in range(0,3):
				# train 4 times:
				MET_AG_main(test_subject=test_subject, shuffle=True, hyperparameters=hyperparameters)
				clean(directory)    
		
		elif args.testSubject == '0' or args.testSubject is None: #10-fold cv
			testList = ['S1', 'S2', 'S3', 'S4', 'S5',
						'S6', 'S7', 'S8', 'S9', 'S10']

			for test_subject in testList:
				MET_AG_main(test_subject=test_subject, hyperparameters=hyperparameters)
				clean(directory)

		elif int(args.testSubject) <= 10 and int(args.testSubject) >= 1:
			MET_AG_main(test_subject='S'+args.testSubject, hyperparameters=hyperparameters, ftune=args.fineTune, modelname=args.modelname, saveModel=args.saveModel)
			clean(directory)

	elif args.function == 'count_MET_main':   
		if args.shuffle:
			test_subject = 'S10' #dummy

			for i in range(0,3):
				# train 4 times:
				count_MET_main(test_subject=test_subject, shuffle=True, hyperparameters=hyperparameters)
				clean(directory)

		elif args.testSubject == '0' or args.testSubject is None: #10-fold cv
			testList = ['S1', 'S2', 'S3', 'S4', 'S5',
						'S6', 'S7', 'S8', 'S9', 'S10']

			hyperparameters[7] = [1., .001, .1]
			for test_subject in testList:
				count_MET_main(test_subject=test_subject, hyperparameters=hyperparameters)
				clean(directory)

		elif int(args.testSubject) <= 10 and int(args.testSubject) >= 1:
			count_MET_main(test_subject='S'+args.testSubject, hyperparameters=hyperparameters, ftune=args.fineTune, modelname=args.modelname, saveModel=args.saveModel)
			clean(directory)
	
	elif args.function == 'count_main':   
		if args.shuffle:
			test_subject = 'S10' #dummy

			for i in range(0,3):
				# train 4 times:
				count_main(test_subject=test_subject, shuffle=True, hyperparameters=hyperparameters)
				clean(directory) 

		elif args.testSubject == '0' or args.testSubject is None: #10-fold cv
			testList = ['S1', 'S2', 'S3', 'S4', 'S5',
						'S6', 'S7', 'S8', 'S9', 'S10']

			for test_subject in testList:
				count_main(test_subject=test_subject, hyperparameters=hyperparameters)
				clean(directory)

		elif int(args.testSubject) <= 10 and int(args.testSubject) >= 1:
			count_main(test_subject='S'+args.testSubject, hyperparameters=hyperparameters, ftune=args.fineTune, modelname=args.modelname, saveModel=args.saveModel)
			clean(directory)

	elif args.function == 'countOnly':
		if args.shuffle:
			test_subject = 'S10' #dummy

			for i in range(0,3):
				# train 4 times:
				countOnly(test_subject=test_subject, shuffle=True, hyperparameters=hyperparameters)
				clean(directory)  

		elif args.testSubject == '0' or args.testSubject is None: #10-fold cv
			testList = ['S1', 'S2', 'S3', 'S4', 'S5',
						'S6', 'S7', 'S8', 'S9', 'S10']

			for test_subject in testList:
				countOnly(test_subject=test_subject, hyperparameters=hyperparameters)
				clean(directory)

		elif int(args.testSubject) <= 10 and int(args.testSubject) >= 1:
			countOnly(test_subject='S'+args.testSubject, hyperparameters=hyperparameters, ftune=args.fineTune, modelname=args.modelname, saveModel=args.saveModel)
			clean(directory)

	else:
		ValueError("Arguments are not correct. Get help by --help")