# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import confusion_matrix
#
# from keras.wrappers.scikit_learn import KerasClassifier
#
# from sklearn.pipeline import Pipeline
# import os
# import PIL
import pandas

# load dataset
dataframe = pandas.read_csv("ML_datase_book_v1.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:3].astype(float)
Y = dataset[:,3]



# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, dummy_y, test_size=0.40, random_state=1)
# n_train = 10000
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = dummy_y[:n_train], dummy_y[n_train:]

# # define baseline model
# def baseline_model():
#     model = Sequential()
#     model.add(Dense(4, input_dim=3, activation='relu'))
#     model.add(Dense(units=32, activation='elu'))
#     model.add(Dense(units=64, activation='elu'))
#     model.add(Dense(units=128, activation='elu'))
#     model.add(Dense(units=128, activation='elu'))
#     model.add(Dense(units=256, activation='elu'))
#     model.add(Dense(units=512, activation='elu'))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# define model
model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
# model.add(Dense(3, activation='sigmoid'))
# opt = SGD(lr=0.01, momentum=0.9)
model.add(Dense(3, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
# history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=200, verbose=0)
predictions = model.predict(X_validation)
# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=1)
_, test_acc = model.evaluate(X_validation, Y_validation, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='Validation_loss')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='Validation_loss')
pyplot.legend()
pyplot.show()
# cm = confusion_matrix(y_true=Y_validation[:, 0], y_pred=predictions)
# cm_plot_labels = ['Normal', 'MIMA', 'GPS Spoofing']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
# Plot non-normalized confusion matrix
# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# plot_confusion_matrix(model, X_validation, Y_validation)
# plt.show()
# estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=4, verbose=1)
# kfold = KFold(n_splits=2, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# define the keras model
# model = Sequential()
# model.add(Dense(12, input_dim=3, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(3, activation='sigmoid'))
# # compile the keras model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit the keras model on the dataset
# model.fit(X, dummy_y, epochs=150, batch_size=10, verbose=0)
# # evaluate the keras model
# _, accuracy = model.evaluate(X, dummy_y)
# print('Accuracy: %.2f' % (accuracy*100))
# # make class predictions with the model
# predictions = model.predict_classes(X)
# # summarize the first 5 cases
# for i in range(5):
# 	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], dummy_y[i]))