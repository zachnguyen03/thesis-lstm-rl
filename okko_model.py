import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import re
import matplotlib.pyplot as plt
import pandas as pd
​
​
# Read file
filename = 'Dataset/wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
raw_text = re.sub(r'[^a-z0-9.,]', ' ', raw_text)
​
#Stopwords removal
import nltk
nltk.download('stopwords')
nltk.download('word_tokenize')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(raw_text)
raw_text = ' '.join([w for w in word_tokens if w not in stop_words or w != ' '])
​
# List appeared characters in the text
chars = sorted(list(set(raw_text)))
​
# Map integer to character and vice versa
int_to_char = dict((i,c) for i,c in enumerate(chars))
char_to_int = dict((c,i) for i,c in enumerate(chars))
num_classes = len(int_to_char)
n_chars = len(raw_text)
n_vocab = len(chars)
​
# Calculate patterns in the text
seq_length = 40
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
​
n_patterns = len(dataX)
print("Total patterns: ", n_patterns)
​
​
# Data transformation for Keras
#X = np.reshape(dataX, [n_patterns, seq_length, 1])
#X = X / float(n_vocab)
​
X = tf.keras.utils.to_categorical(dataX, num_classes=num_classes)
y = tf.keras.utils.to_categorical(dataY, num_classes=num_classes)
​
## NEW MODEL BEGINS
​
import scipy,scipy.io
import os,sys
import numpy as np
#import keras
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, TimeDistributed, add,Conv1D,Conv2D,MaxPooling2D,MaxPooling1D, AveragePooling1D, Reshape
from keras.models import Model, load_model
import keras.backend as K
from keras.engine import Layer
from keras import regularizers
import tensorflow as tf
​
​
conv_length = [2,2,2,2,2,2]
conv_dilation = [1,2,4,8,16,32] # at 8kHz, last layer is 128 ms
pooling_length = [1,1,1,1,1,1]

# Regularization terms: increase these if overfitting is a problem
# (or you get NaNs due to exploding gradients)
actreg      = 0.000000000001
relu_actreg = 0.000000000001
​
# Network dimensions
n_channels = 64 # Convolutional filter channels per layer
​
​
# Define the WaveNet layers (note: rewriting in loops is more compact, here just explicit )
text_input = tf.keras.layers.Input(shape=(X.shape[1:]))
​
encoder0 = tf.keras.layers.Conv1D(n_channels,(conv_length[0]),dilation_rate=conv_dilation[0],activation='sigmoid',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder1 = tf.keras.layers.Conv1D(n_channels,(conv_length[1]),dilation_rate=conv_dilation[1],activation='sigmoid',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder2 = tf.keras.layers.Conv1D(n_channels,(conv_length[2]),dilation_rate=conv_dilation[2],activation='sigmoid',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder3 = tf.keras.layers.Conv1D(n_channels,(conv_length[3]),dilation_rate=conv_dilation[3],activation='sigmoid',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder4 = tf.keras.layers.Conv1D(n_channels,(conv_length[4]),dilation_rate=conv_dilation[4],activation='sigmoid',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder5 = tf.keras.layers.Conv1D(n_channels,(conv_length[5]),dilation_rate=conv_dilation[5],activation='sigmoid',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
​
encoder0_tanh = tf.keras.layers.Conv1D(n_channels,(conv_length[0]),dilation_rate=conv_dilation[0],activation='tanh',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder1_tanh = tf.keras.layers.Conv1D(n_channels,(conv_length[1]),dilation_rate=conv_dilation[1],activation='tanh',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder2_tanh = tf.keras.layers.Conv1D(n_channels,(conv_length[2]),dilation_rate=conv_dilation[2],activation='tanh',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder3_tanh = tf.keras.layers.Conv1D(n_channels,(conv_length[3]),dilation_rate=conv_dilation[3],activation='tanh',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder4_tanh = tf.keras.layers.Conv1D(n_channels,(conv_length[4]),dilation_rate=conv_dilation[4],activation='tanh',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
encoder5_tanh = tf.keras.layers.Conv1D(n_channels,(conv_length[5]),dilation_rate=conv_dilation[5],activation='tanh',padding='causal',activity_regularizer=tf.keras.regularizers.l2(actreg))
​

​skip_scaler0 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
skip_scaler1 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
skip_scaler2 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
skip_scaler3 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
skip_scaler4 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
skip_scaler5 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
​
res_scaler0 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
res_scaler1 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
res_scaler2 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
res_scaler3 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
res_scaler4 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
res_scaler5 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_channels,activation='linear'))
​
summer = tf.keras.layers.Add()
multiplier = tf.keras.layers.Multiply()
concatenator = tf.keras.layers.Concatenate()
​
​
# Put the network together
​
l0_skip = kip_scaler0(multiplier([encoder0(text_input),encoder0_tanh(text_input)]))
l0_res = res_scaler0(l0_skip)
l1_skip = skip_scaler1(multiplier([encoder1(l0_res),encoder1_tanh(l0_res)]))
l1_res = res_scaler1(summer([l0_res,l1_skip]))
l2_skip = skip_scaler2(multiplier([encoder2(l1_res),encoder2_tanh(l1_res)]))
l2_res = res_scaler2(summer([l1_res,l2_skip]))
l3_skip = skip_scaler3(multiplier([encoder3(l2_res),encoder3_tanh(l2_res)]))
l3_res = res_scaler3(summer([l2_res,l3_skip]))
l4_skip = skip_scaler4(multiplier([encoder4(l3_res),encoder4_tanh(l3_res)]))
l4_res = res_scaler4(summer([l3_res,l4_skip]))
l5_skip = skip_scaler5(multiplier([encoder5(l4_res),encoder5_tanh(l4_res)]))
l5_res = res_scaler5(summer([l4_res,l5_skip]))
​
​
# Regular wavenet: outputting from each layer
convstack_out = summer([l0_skip,l1_skip])
convstack_out = summer([convstack_out,l2_skip])
convstack_out = summer([convstack_out,l3_skip])
convstack_out = summer([convstack_out,l4_skip])
convstack_out = summer([convstack_out,l5_skip])
​
convstack_out = tf.keras.layers.MaxPooling1D(40)(convstack_out)
convstack_out = tf.keras.layers.Reshape([n_channels])(convstack_out)
postnet = tf.keras.layers.Dense(n_channels,activation='relu')
softmax = tf.keras.layers.Dense(y.shape[-1],activation='softmax')
​
model_output = softmax(postnet(convstack_out))
​
model = tf.keras.models.Model(inputs=text_input,outputs=model_output)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))

filepath = "./Weights/weight_lstm2-okko2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history3 = model.fit(X,y,shuffle=True, epochs=100,batch_size=128, callbacks=callbacks_list)
model.load_weights(filepath)
​
## NEW MODEL ENDS
​
# Generate Text using custom Text Generation function
def text_generation(length, diversity):
    # Pick random seed from input text
    outputted_text = ''
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    # Loop through each pattern and predict using LSTM Model and print out the results
    for i in range(length):
        patternX = tf.keras.utils.to_categorical(pattern, num_classes=num_classes)
        patternX = np.reshape(patternX, (1, patternX.shape[0], patternX.shape[1]))
        prediction = model.predict(patternX, verbose=0)
#        index = np.argmax(prediction)
        index = sample(prediction, diversity)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        outputted_text += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return outputted_text
​
​
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds.T, 1)
    return np.argmax(probas)
​
outputted_text = text_generation(1000, 0.15)
​
​
# Save model
#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
#
#
## returns a compiled model
## identical to the previous one
model = tf.keras.models.load_model('my_model.h5')