import csv
import os
import numpy as np
from sklearn.utils import shuffle



def csv_parse(filename, path_to_dir):
	print("\nReading file: {}".format(filename))
	fullpath = path_to_dir + '/' + filename # str format
	headerline = 2
	raw_list = []
	headerlist = []

	with open(fullpath) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		i=0

		for row in readCSV:
			if i < headerline: # headerline
				headerlist.extend([row])
				i += 1
			else:
				raw_list.extend([row])
		raw_array = np.asarray(raw_list)
	return headerlist, raw_array


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def get_dataset(path_to_dir):
	data = find_csv_filenames(path_to_dir)
	data = [ file for file in data if 'dataset' in file]
	epochslist = find_csv_filenames(path_to_dir, 'epochs.csv')
	no_data = len(epochslist)
	d_list = []
	e_list = []
	for i in range(1, int(no_data+1)):
		for d in data:
			name = 'S'+str(i)+'.'
			if (name in d) & ('epochs' not in d):
				d_list.append(d)
				e_list.append(d[:-4]+'_epochs.csv')
	return (d_list, e_list)




def main(path_to_dir, test='S10'):
	test_no = int(test[1:])-1
	filenames, epochnames = get_dataset(path_to_dir)
	testfile = [filenames[test_no]]
	testepoch = [epochnames[test_no]]
	del filenames[test_no]
	del epochnames[test_no]

	X_train , X_test, y_train, y_test = None, None, None, None
	
	for file in filenames:
		[_, data] = csv_parse(file, path_to_dir)
		if X_train is None:
			X_train = np.reshape(data[:, 6:15], (-1, 200, 9))
		else:
			X_train = np.concatenate((X_train, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
	
	"""
	[_, data] = csv_parse(filenames[0])
	X = np.reshape(data[:, 6:15], (-1, 200, 9))
	[_, data] = csv_parse(filenames[1])
	X = np.concatenate((X, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
	[_, data] = csv_parse(filenames[2])
	X = np.concatenate((X, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
	[_, data] = csv_parse(filenames[3])
	X_train = np.concatenate((X, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
	"""
	epochs, kcal_MET = None, None
	for epoch in epochnames:
		[_, data] = csv_parse(epoch, path_to_dir)
		eps = data[:2,8:11]
		km = data[:2,11:13]
		y_ = data[:2,8]
		for row in data[:,:13]:
			if row[8]:
				eps = np.append(eps, [row[8:11]], axis=0)
				km = np.append(km, [row[11:]], axis=0)
				if row[2] == "SITTING":
					y_ = np.concatenate((y_, [0]), axis=0)
				elif row[2] == "LYING":
					y_ = np.concatenate((y_, [1]), axis=0)
				elif row[2] == "STANDING":
					y_ = np.concatenate((y_, [2]), axis=0)
				elif row[2] == "WALKING":
					y_ = np.concatenate((y_, [3]), axis=0)
				elif row[2] == "RUNNING":
					y_ = np.concatenate((y_, [4]), axis=0)
				elif row[2] == "DOWNSTAIRS":
					y_ = np.concatenate((y_, [5]), axis=0)
				elif row[2] == "UPSTAIRS":
					y_ = np.concatenate((y_, [6]), axis=0)
		
		if y_train is None:
			y_train = np.reshape(y_[2:], (-1, 1))
		else:
			y_train = np.concatenate((y_train, np.reshape(y_[2:], (-1, 1))), axis=0)
		y_ = None

		if epochs is None:
			epochs = np.reshape(eps[2:,:], (-1, 3))
		else:
			epochs = np.concatenate((epochs, np.reshape(eps[2:,:], (-1, 3))), axis=0)
		if kcal_MET is None:
			kcal_MET = np.reshape(km[2:,:], (-1, 2))
		else:
			kcal_MET = np.concatenate((kcal_MET, np.reshape(km[2:,:], (-1, 2))), axis=0)


	test_epochs, test_kcal_MET = None, None
	for file in testfile:
		[_, data] = csv_parse(file, path_to_dir)
		if X_test is None:
			X_test = np.reshape(data[:, 6:15], (-1, 200, 9))
		else:
			X_test = np.concatenate((X_test, np.reshape(data[:, 6:15], (-1, 200, 9))), axis=0)

	for epoch in testepoch:
		[_, data] = csv_parse(epoch, path_to_dir)
		test_eps = data[:2,8:11]
		test_km = data[:2,11:13]
		y_ = data[:2,8]
		for row in data[:,:13]:
			if row[8]:
				test_eps = np.append(test_eps, [row[8:11]], axis=0)
				test_km = np.append(test_km, [row[11:]], axis=0)
				if row[2] == "SITTING":
					y_ = np.concatenate((y_, [0]), axis=0)
				elif row[2] == "LYING":
					y_ = np.concatenate((y_, [1]), axis=0)
				elif row[2] == "STANDING":
					y_ = np.concatenate((y_, [2]), axis=0)
				elif row[2] == "WALKING":
					y_ = np.concatenate((y_, [3]), axis=0)
				elif row[2] == "RUNNING":
					y_ = np.concatenate((y_, [4]), axis=0)
				elif row[2] == "DOWNSTAIRS":
					y_ = np.concatenate((y_, [5]), axis=0)
				elif row[2] == "UPSTAIRS":
					y_ = np.concatenate((y_, [6]), axis=0)

		if y_test is None:
			y_test = np.reshape(y_[2:], (-1, 1))
		else:
			y_test = np.concatenate((y_test, np.reshape(y_[:2], (-1, 1))), axis=0)
		y_ = None

		if test_epochs is None:
			test_epochs = test_eps[2:,:]
		else:
			test_epochs = np.concatenate((test_epochs, test_eps[2:,:]), axis=0)
		if test_kcal_MET is None:
			test_kcal_MET = test_km[2:,:]
		else:
			test_kcal_MET = np.concatenate((test_kcal_MET, test_km[2:,:]), axis=0)

	


	"""
	outputs:
	X_train (7770, 200, 9)
	y_train (7770, 1)
	X_test (630, 200, 9)
	y_test (630, 1)
	epochs (630, 3)
	test_epochs (630, 3)
	kcal_MET (630, 2)
	test_kcal_MET (630, 2)
	"""

	#shuffle training set
	y_train = np.concatenate((y_train.astype(np.float), epochs.astype(np.float), kcal_MET.astype(np.float)), axis=1)
	X_train, y_train = shuffle(X_train.astype(np.float), y_train)
	epochs = y_train[:,1:4]   # extract count values
	kcal_MET = y_train[:, 4:] # extract kcal & MET values
	y_train = y_train[:,0].reshape(-1, 1)   # extract class labels

	return X_train, y_train, epochs, X_test.astype(np.float), y_test.astype(np.float), test_epochs.astype(np.float), kcal_MET, test_kcal_MET.astype(np.float)
