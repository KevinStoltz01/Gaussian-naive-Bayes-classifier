import os 
import pandas as pd
import numpy as np
import math
import sklearn
from sklearn import metrics

#read in data with column names labeled to make sense of what I'm seeing 
f_pd = pd.read_csv('/Users/kevinstoltz/Desktop/spambase.data',
               names = ['word_freq_make',
               'word_freq_address',
               'word_freq_all',
               'word_freq_3d',
               'word_freq_our',
               'word_freq_over',
               'word_freq_remove',
               'word_freq_internet',
               'word_freq_order',
               'word_freq_mail',
               'word_freq_receive',
               'word_freq_will',
               'word_freq_people',
               'word_freq_report',
               'word_freq_addresses',
               'word_freq_free',
               'word_freq_business',
               'word_freq_email',
               'word_freq_you',
               'word_freq_credit',
               'word_freq_your',
               'word_freq_font',
               'word_freq_000',
               'word_freq_money',
               'word_freq_hp',
               'word_freq_hpl',
               'word_freq_george',
               'word_freq_650',
               'word_freq_lab',
               'word_freq_labs',
               'word_freq_telnet',
               'word_freq_857',
               'word_freq_data',
               'word_freq_415',
               'word_freq_85',
               'word_freq_technology',
               'word_freq_1999',
               'word_freq_parts',
               'word_freq_pm',
               'word_freq_direct',
               'word_freq_cs',
               'word_freq_meeting',
               'word_freq_original',
               'word_freq_project',
               'word_freq_re',
               'word_freq_edu',
               'word_freq_table',
               'word_freq_conference',
               'char_freq_;',
               'char_freq_(',
               'char_freq_[',
               'char_freq_!',
               'char_freq_$',
               'char_freq_#',
               'capital_run_length_average',
               'capital_run_length_longest',
               'capital_run_length_total',
               'class (1=spam, 0=not spam)'])
f_pd


#set up a variable called random that will allow me to randomly sample from data
#the < 0.5 changes this to a boolean expression where any of the random numbers just generated that are < 0.5 return TRUE
random = np.random.rand(len(f_pd)) < 0.5

#create training and test set 
#the documentation for how the boolean mask array indexing works is here... 
#https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html (about half way down the page)
train = f_pd[random]
test = f_pd[~random]

#represent data in a numpy array 
train = np.array(train)
test = np.array(test)

#dims check on training and test set
print("train shape:", train.shape)
print("test shape:", test.shape)

#print some relavent info about the training and test sets 
train_no_spam = np.sum(train[:,57])
test_no_spam = np.sum(test[:,57])

print("train spam instances = ", train_no_spam)
print("test spam instances = ", test_no_spam)

train_fract_spam = train_no_spam/train.shape[0] 
test_fract_spam = test_no_spam/test.shape[0]

train_fract_not_spam = 1-train_fract_spam
test_fract_not_spam = 1-test_fract_spam

print("train fraction of spam = ", train_fract_spam)
print("test fraction of spam = ", test_fract_spam)
print("train_fract_not_spam =", train_fract_not_spam)
print("test_fract_not_spam = ", test_fract_not_spam)

#create labels from test set for model evaluation
labels = test[:,57]
len(labels)

#remove labels form the test set 
test = test[:, 0:57]
test.shape

#split training set into examples that contain spam and examples that don't contain spam 
train_spam = train[np.where(train[:,57] == 1)]
train_not_spam = train[np.where(train[:,57] == 0)]

print("train_spam shape = ",train_spam.shape)
print("train_not_spam shape = ",train_not_spam.shape)

#compute mean and stdev given spam
train_spam_mean = np.mean(train_spam, axis=0)
train_spam_stdev = np.std(train_spam, axis=0)

#trim off labels 
train_spam_mean = train_spam_mean[0:57]
train_spam_stdev = train_spam_stdev[0:57]

#add 0.0001 to any instances where stdev = 0 
for i in range(len(train_spam_stdev)):
    if train_spam_stdev[i] == 0:
        train_spam_stdev[i] = train_spam_stdev[i] + 0.0001

#reshape
train_spam_mean = np.reshape(train_spam_mean, (1,57))
train_spam_stdev = np.reshape(train_spam_stdev, (1, 57))

#print dims sanity check 
print("train_spam_mean shape = ", train_spam_mean.shape)
print("train_spam_stdev shape = ", train_spam_stdev.shape)


#mean and stdev given not spam (this section same as above, just with not spam class)
train_not_spam_mean = np.mean(train_not_spam, axis=0)
train_not_spam_stdev = np.std(train_not_spam, axis=0)

train_not_spam_mean = train_not_spam_mean[0:57]
train_not_spam_stdev = train_not_spam_stdev[0:57]

for i in range(len(train_not_spam_stdev)):
    if train_not_spam_stdev[i] == 0:
        train_not_spam_stdev[i] = train_not_spam_stdev[i] + 0.0001

train_not_spam_mean = np.reshape(train_not_spam_mean, (1,57))
train_not_spam_stdev = np.reshape(train_not_spam_stdev, (1, 57))

print("train_not_spam_mean shape = ", train_spam_mean.shape)
print("train_not_spam_stdev shape = ", train_spam_stdev.shape)

#compute conditional probabilities for test set given spam and not spam  

#spam class

#Gaussian function in parts 
#(x-mu)^2
xMinusMuSqr_spam = (test - train_spam_mean)**2
#2(sigma^2)
twoTimesSigSqr_spam = 2*(train_spam_stdev**2)
#1/(sqrt(2pi)*sigma)
gauss_1_spam = (1/(math.sqrt(2*math.pi)*train_spam_stdev))
#e^-((x-mu)^2/2*variance)
gauss_2_spam = np.exp(-(xMinusMuSqr_spam/twoTimesSigSqr_spam))
#the full Gaussian function 
full_gauss_spam = gauss_1_spam*gauss_2_spam

#predictions for spam class 
pred_spam = np.log(train_fract_spam)+np.sum(np.log(full_gauss_spam), axis=1)

#not spam class
xMinusMuSqr_not_spam = (test - train_not_spam_mean)**2
twoTimesSigSqr_not_spam = 2*(train_not_spam_stdev**2)

gauss_1_not_spam = (1/(math.sqrt(2*math.pi)*train_not_spam_stdev))
gauss_2_not_spam = np.exp(-(xMinusMuSqr_not_spam/twoTimesSigSqr_not_spam))
full_gauss_not_spam = gauss_1_not_spam*gauss_2_not_spam
        
#predictions for not spam class
pred_not_spam = np.log(train_fract_not_spam)+np.sum(np.log(full_gauss_not_spam), axis=1)

#reshaping the predictions 
pred_spam = pred_spam.reshape(len(labels),1)
pred_not_spam = pred_not_spam.reshape(len(labels),1)

#dims sanity check 
print('pred_spam shape = ', pred_spam.shape)
print('pred_not_spam shape = ', pred_not_spam.shape)

#concatenating the two prediction vectors
pred = np.concatenate((pred_not_spam,pred_spam), axis =1)

#taking argmax to determine class output 
pred = np.argmax(pred, axis=1)

#confusion matrix 
conf = sklearn.metrics.confusion_matrix(labels, pred)
print(conf)

#accuracy, recall, and precision calcs
acc = np.sum(np.diagonal(conf))/np.sum(conf)
prec = conf[1,1]/(conf[0,1]+conf[1,1])
recall = conf[1,1]/(conf[1,1]+conf[1,0])

print("accuracy = ", acc)
print("precision = ", prec)
print("recall = ", recall)

#checking my math with scikit learn
print("sk accuracy = ", sklearn.metrics.accuracy_score(labels, pred))
print("sk precision = ", sklearn.metrics.precision_score(labels, pred))
print("sk recall = ", sklearn.metrics.recall_score(labels, pred))


# I was curious as to how the linearity assumption might have affected my naive bayes results so I set up an MLP to classify the spam data set. 
train_labels = train[:,57]
train = train[:,0:57]
train.shape

import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation 
from keras.optimizers import SGD

model = Sequential()

model.add(Dense(20, input_dim = 57, activation = "sigmoid"))
model.add(Dense(1, activation = "sigmoid"))
  
sgd = SGD(lr = 0.1, momentum = 0.0, decay=0.0)    
model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
          
model.fit(train, train_labels, epochs = 10, batch_size = 3)

keras_pred = model.predict(test)
keras_pred = np.where(keras_pred > 0.5, 1, 0)

labels = labels.astype(int)

conf_keras = sklearn.metrics.confusion_matrix(labels, keras_pred)
print(conf_keras)
print("keras accuracy = ", sklearn.metrics.accuracy_score(labels, keras_pred))
print("keras precision = ", sklearn.metrics.precision_score(labels, keras_pred))
print("keras recall = ", sklearn.metrics.recall_score(labels, keras_pred))

